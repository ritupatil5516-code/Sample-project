# core/orchestrator/execute.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datetime import datetime

# --- RAG lane (LangChain + LlamaIndex adapters; must exist in your project) ---
try:
    from core.retrieval.rag_chain import (
        unified_rag_answer,
        account_rag_answer,
        knowledge_rag_answer,
    )
except Exception:
    # Safe fallback if RAG isn’t wired yet
    def unified_rag_answer(question, session_id, account_id=None, k=6):
        return {"answer": "", "sources": [], "error": "RAG not available"}
    account_rag_answer = unified_rag_answer
    knowledge_rag_answer = unified_rag_answer


# ============================= config helpers =============================

def _read_app_cfg(path: str = "config/app.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _accounts_dir(cfg: Dict[str, Any]) -> Path:
    base = ((cfg.get("data") or {}).get("accounts_dir") or "src/api/contextApp/data/customer_data").strip()
    return Path(base)

# ============================= JSON loaders ==============================

def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # Be forgiving if a file is saved with ANSI; avoid crashing.
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return None

def _load_domain_rows_for_account(accounts_dir: Path, account_id: Optional[str], domain: str) -> List[Dict[str, Any]]:
    """
    Returns a list of dict rows for the domain under account_id.
    account_summary may be a single dict -> converted into [dict].
    """
    if not account_id:
        return []
    base = accounts_dir / account_id
    fname = {
        "transactions": "transactions.json",
        "payments": "payments.json",
        "statements": "statements.json",
        "account_summary": "account_summary.json",
    }.get(domain, "")
    if not fname:
        return []
    raw = _load_json(base / fname)
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    return []

# ============================== DSL helpers ==============================

def _iso(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts))
        except Exception:
            return None
    s = str(ts)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _latest_key(row: Dict[str, Any]) -> Tuple:
    for k in ("postedDateTime", "transactionDateTime", "paymentPostedDateTime", "closingDateTime", "openingDateTime", "date"):
        dt = _iso(row.get(k))
        if dt:
            return (dt,)
    return (datetime.min,)

def _matches_where(row: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    if not where:
        return True
    for k, v in where.items():
        rv = row.get(k)
        if isinstance(v, str):
            if str(rv).lower().find(v.lower()) < 0:
                return False
        else:
            if rv != v:
                return False
    return True

def _get_by_dotted(obj: Any, path: str):
    """
    Supports dotted + bracketed paths: persons[0].ownershipType
    """
    if not isinstance(path, str):
        return None
    cur = obj
    path = path.replace("[", ".").replace("]", "")
    for part in [p for p in path.split(".") if p]:
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except Exception:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur

# ---------- DSL ops (domain-agnostic) ----------

def op_get_field(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    if not rows:
        return {"value": None, "row": None}
    # If there are many rows (e.g., transactions), pick the latest row by timestamp
    if len(rows) > 1:
        row = max(rows, key=_latest_key)
    else:
        row = rows[0]
    return {"value": _get_by_dotted(row, field), "row": row}

def op_find_latest(rows: List[Dict[str, Any]], where: Optional[Dict[str, Any]] = None, key: Optional[str] = None) -> Dict[str, Any]:
    cand = [r for r in rows if _matches_where(r, where)]
    if not cand:
        return {"row": None, "count": 0}
    if key:
        # use explicit key if provided
        def _k(r):
            v = _get_by_dotted(r, key)
            dt = _iso(v)
            return dt or datetime.min
        latest = max(cand, key=_k)
    else:
        latest = max(cand, key=_latest_key)
    return {"row": latest, "count": len(cand)}

def op_sum_where(rows: List[Dict[str, Any]], where: Optional[Dict[str, Any]] = None, value_path: str = "amount") -> Dict[str, Any]:
    cand = [r for r in rows if _matches_where(r, where)]
    total = 0.0
    for r in cand:
        v = _get_by_dotted(r, value_path)
        try:
            total += float(v or 0.0)
        except Exception:
            pass
    return {"total": total, "count": len(cand)}

def op_topk_by_sum(rows: List[Dict[str, Any]], key_field: str, k: int = 5, where: Optional[Dict[str, Any]] = None, amount_field: str = "amount") -> Dict[str, Any]:
    cand = [r for r in rows if _matches_where(r, where)]
    agg: Dict[str, float] = {}
    for r in cand:
        key_val = _get_by_dotted(r, key_field)
        if key_val is None:
            key_val = "UNKNOWN"
        try:
            amt = float(_get_by_dotted(r, amount_field) or 0.0)
        except Exception:
            amt = 0.0
        agg[str(key_val)] = agg.get(str(key_val), 0.0) + amt
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(k or 5))]
    return {"top": [{"key": t[0], "total": t[1]} for t in top], "count": len(cand)}

def op_list_where(rows: List[Dict[str, Any]], where: Optional[Dict[str, Any]] = None, limit: int = 15) -> Dict[str, Any]:
    cand = [r for r in rows if _matches_where(r, where)]
    return {"items": cand[: max(1, int(limit or 15))], "count": len(cand)}

def op_semantic_search(rows: List[Dict[str, Any]], query: str, k: int = 5) -> Dict[str, Any]:
    """
    Lightweight, dependency-free semantic-ish search:
    - ranks by keyword overlap on common text fields (merchantName, description, category)
    If you already have a transactions FAISS index, replace this with your `faiss_registry.query_index` call.
    """
    if not query:
        return {"hits": []}
    q = query.lower()
    fields = ("merchantName", "description", "displayTransactionType", "category")
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        txt = " ".join(str(r.get(f) or "") for f in fields).lower()
        score = 0.0
        for term in q.split():
            if term in txt:
                score += 1.0
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    hits = []
    for s, r in scored[: max(1, int(k or 5))]:
        hits.append({"score": s, "payload": r, "text": f"{r.get('merchantName') or r.get('description') or ''}"})
    return {"hits": hits}

# =============================== EXECUTE =================================

def execute_calls(calls: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the plan's calls and returns a results dict keyed like:
      "<domain>.<capability>[i]" -> payload
    Context accepts:
      - app_yaml: path to config
      - session_id: str (for RAG memory)
      - account_id: default account if args omit it
    """
    cfg = _read_app_cfg(context.get("app_yaml") or "config/app.yaml")
    accounts_dir = _accounts_dir(cfg)
    session_id = context.get("session_id") or "default-session"

    out: Dict[str, Any] = {}

    for i, call in enumerate(calls or []):
        dom = str(call.get("domain_id", "")).strip().lower().replace("-", "_")
        cap = str(call.get("capability", "")).strip().lower().replace("-", "_")
        args = dict(call.get("args") or {})
        key = f"{dom}.{cap}[{i}]"

        # ---------------- RAG lane ----------------
        if dom == "rag":
            if cap == "unified_answer":
                res = unified_rag_answer(
                    question=context.get("question") or "",
                    session_id=session_id,
                    account_id=args.get("account_id") or context.get("account_id"),
                    k=int(args.get("k", 6)),
                )
            elif cap == "account_answer":
                res = account_rag_answer(
                    question=context.get("question") or "",
                    session_id=session_id,
                    account_id=args.get("account_id") or context.get("account_id"),
                    k=int(args.get("k", 6)),
                )
            elif cap == "knowledge_answer":
                res = knowledge_rag_answer(
                    question=context.get("question") or "",
                    session_id=session_id,
                    k=int(args.get("k", 6)),
                )
            else:
                res = {"error": f"Unknown RAG capability: {cap}"}
            out[key] = res
            continue

        # ---------------- DSL lane ----------------
        # For any other domain, we assume it's one of:
        # transactions | payments | statements | account_summary
        account_id = args.get("account_id") or context.get("account_id")
        rows = _load_domain_rows_for_account(accounts_dir, account_id, dom)

        try:
            if cap == "get_field":
                field = args.get("field") or args.get("path") or ""
                res = op_get_field(rows, field=field)
            elif cap == "find_latest":
                res = op_find_latest(rows, where=args.get("where"), key=args.get("key"))
            elif cap == "sum_where":
                res = op_sum_where(rows, where=args.get("where"), value_path=args.get("value_path") or "amount")
            elif cap == "topk_by_sum":
                res = op_topk_by_sum(
                    rows,
                    key_field=args.get("key_field") or "merchantName",
                    k=int(args.get("k", 5)),
                    where=args.get("where"),
                    amount_field=args.get("amount_field") or "amount",
                )
            elif cap == "list_where":
                res = op_list_where(rows, where=args.get("where"), limit=int(args.get("limit", 15)))
            elif cap == "semantic_search":
                # Default: lightweight keyword scoring on local rows
                # If you have FAISS for transactions, you can special-case dom=='transactions' and call that instead.
                res = op_semantic_search(rows, query=args.get("query") or "", k=int(args.get("k", 5)))
            else:
                res = {"error": f"Unknown capability: {cap}"}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}

        out[key] = res

    return out