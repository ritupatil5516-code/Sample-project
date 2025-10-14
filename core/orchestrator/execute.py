# core/orchestrator/execute.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta, timezone

# deterministic loaders / calculators
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

from domains.transactions import calculator as txn_calc
from domains.payments import calculator as pay_calc
from domains.statements import calculator as stmt_calc
from domains.account_summary import calculator as acct_calc

# vector search for semantic ops (transactions)
from core.index.faiss_registry import query_index, Embedder

# RAG lane with conversational memory
from core.retrieval.rag_chain import (
    unified_rag_answer,     # account JSONs + knowledge
    account_rag_answer,     # account JSONs only (optional)
    knowledge_rag_answer,   # knowledge only (optional)
)
def _normalize_calls(calls: Any) -> List[Dict[str, Any]]:
    """
    Coerce plan['calls'] into a list[dict] with keys:
      - domain_id
      - capability
      - args (dict)
    Accepts items that are strings (JSON or op-name), dicts with alternate keys, etc.
    Silently drops unparseable items.
    """
    if calls is None:
        return []
    if isinstance(calls, dict):
        calls = [calls]
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(calls):
        # Allow raw strings: JSON, or a bare op like "unified_answer"
        if isinstance(c, str):
            s = c.strip()
            if not s:
                continue
            if s.startswith("{") and s.endswith("}"):
                try:
                    c = json.loads(s)
                except Exception:
                    continue
            else:
                # assume a rag op; default domain to 'rag'
                c = {"domain_id": "rag", "capability": s, "args": {}}

        if not isinstance(c, dict):
            continue

        # Tolerate alternate key names from planner
        dom = c.get("domain_id") or c.get("domain") or c.get("domain_key") or ""
        cap = c.get("capability") or c.get("op") or c.get("operation") or c.get("action") or ""
        args = c.get("args") or c.get("params") or {}

        # Final shape
        c = {
            "domain_id": str(dom).strip().lower().replace("-", "_"),
            "capability": str(cap).strip().lower().replace("-", "_"),
            "args": args if isinstance(args, dict) else {}
        }
        out.append(c)
    return out

# ----------------- helpers -----------------

from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone

_TS_KEYS = (
    "postedDateTime", "transactionDateTime", "paymentPostedDateTime",
    "paymentDateTime", "closingDateTime", "openingDateTime", "date",
)

def _pick_latest_row(rows: List[dict]) -> Optional[dict]:
    def _ts(row: dict) -> Optional[datetime]:
        for k in _TS_KEYS:
            v = row.get(k)
            if isinstance(v, str) and v:
                try:
                    return datetime.fromisoformat(v.replace("Z", "+00:00"))
                except Exception:
                    continue
        return None
    best = None
    best_ts = None
    for r in rows:
        t = _ts(r)
        if t is not None and (best_ts is None or t > best_ts):
            best, best_ts = r, t
    return best or (rows[-1] if rows else None)

def _get_by_path(obj: Any, path: str) -> Any:
    """Dot-path extractor, e.g. 'accountStatus' or 'person.0.personName'."""
    if path is None or path == "":
        return obj
    cur: Any = obj
    for token in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(token)]
                continue
            except Exception:
                return None
        if isinstance(cur, dict):
            cur = cur.get(token)
            continue
        # cannot go deeper
        return None
    return cur

def _apply_where(rows: List[dict], where: Dict[str, Any]) -> List[dict]:
    """Very small filter: all conditions must match; supports dot-path keys."""
    if not where:
        return rows
    out = []
    for r in rows:
        ok = True
        for k, v in where.items():
            if _get_by_path(r, k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out

def _within_period(txn: Dict[str, Any], period: str | None) -> bool:
    if not period:
        return True
    if period.upper() == "LAST_12M":
        ts = (txn.get("transactionDateTime") or txn.get("postedDateTime") or txn.get("date"))
        if not ts:
            return False
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        return dt >= cutoff
    return True

def _get_field_from_row(row: Dict[str, Any], field: str) -> Any:
    """
    Supports dot paths and case-insensitive fallback:
      - "currentBalance"
      - "persons[0].accountActivity.rewardsEarned"
    """
    if not field:
        return None
    # try dotted/indexed path
    try:
        cur: Any = row
        for part in field.replace("]", "").split("."):
            if "[" in part:
                key, idx = part.split("[", 1)
                cur = cur.get(key, [])
                cur = cur[int(idx)]
            else:
                cur = cur.get(part)
            if cur is None:
                break
        if cur is not None:
            return cur
    except Exception:
        pass
    # fallback: case-insensitive leaf scan
    f_low = field.lower()
    for k, v in row.items():
        if k.lower() == f_low:
            return v
    return None

def _latest_key(row: Dict[str, Any]) -> str:
    for k in ("transactionDateTime","postedDateTime","paymentPostedDateTime","paymentDateTime","closingDateTime","openingDateTime","date"):
        v = row.get(k)
        if v: return str(v)
    return ""

def _read_app_cfg(path: str) -> Dict[str, Any]:
    import yaml, os
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _build_embedder_from_cfg(cfg: Dict[str, Any]) -> Embedder:
    emb = (cfg.get("embeddings") or {})
    provider = (emb.get("provider") or "openai").lower()
    model    = emb.get("openai_model") or emb.get("model") or "text-embedding-3-large"
    api_base = emb.get("openai_base_url") or emb.get("api_base") or None
    api_key  = __import__("os").getenv(emb.get("openai_api_key_env") or "OPENAI_API_KEY", "")
    return Embedder(provider=provider, model=model, api_key=api_key, api_base=api_base)

# --------------- generic 5-op DSL executors -----------------------------------
def _op_get_field(dom: str, args: Dict[str, Any], ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    get_field for any domain. Arguments:
      - field: dot-path inside the chosen row (required)
      - where: {path: value} exact filters (optional)
      - latest: bool (optional, default False) → if True and domain has many rows,
                choose the latest by timestamp; else choose the first row.
    """
    field = (args.get("field") or "").strip()
    if not field:
        return {"error": "get_field requires 'field' path"}

    latest = bool(args.get("latest", False))
    where  = dict(args.get("where") or {})

    # pick domain rows
    if dom == "transactions":
        rows = list(ds.get("transactions") or [])
    elif dom == "payments":
        rows = list(ds.get("payments") or [])
    elif dom == "statements":
        rows = list(ds.get("statements") or [])
    elif dom == "account_summary":
        # account_summary could be a single dict or list
        val = ds.get("account_summary")
        rows = val if isinstance(val, list) else ([val] if isinstance(val, dict) else [])
    else:
        return {"error": f"Unknown domain for get_field: {dom}"}

    rows = _apply_where(rows, where)

    if not rows:
        return {"error": "No matching rows"}

    row = _pick_latest_row(rows) if latest else rows[0]
    value = _get_by_path(row, field)
    return {"value": value, "row": row}

def _op_find_latest(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    data = datasets.get(domain)
    if isinstance(data, dict):
        return {"value": data, "row": data}
    rows = data or []
    if not rows:
        return {"value": None}
    row = max(rows, key=_latest_key)
    return {"value": row, "row": row}

def _op_sum_where(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    field = (args.get("field") or "amount").strip()
    where = args.get("where") or {}
    rows: List[Dict[str, Any]] = datasets.get(domain) or []
    total = 0.0; cnt = 0
    for r in rows:
        ok = True
        for k, v in where.items():
            if r.get(k) != v:
                ok = False; break
        if ok:
            try:
                total += float(r.get(field) or 0)
                cnt += 1
            except Exception:
                pass
    return {"total": total, "count": cnt}

def _op_topk_by_sum(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    group_key = (args.get("group_by") or "merchantName").strip()
    field = (args.get("field") or "amount").strip()
    k = int(args.get("k", 5))
    rows: List[Dict[str, Any]] = datasets.get(domain) or []
    agg: Dict[str, float] = {}
    for r in rows:
        g = (r.get(group_key) or "UNKNOWN")
        try:
            agg[g] = agg.get(g, 0.0) + float(r.get(field) or 0)
        except Exception:
            pass
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return {"top": [{"key": a, "total": b} for a, b in top]}

def _op_list_where(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    where = args.get("where") or {}
    period = args.get("period")
    rows: List[Dict[str, Any]] = datasets.get(domain) or []
    out = []
    for r in rows:
        if period and not _within_period(r, period):
            continue
        ok = True
        for k, v in where.items():
            if r.get(k) != v:
                ok = False; break
        if ok:
            out.append(r)
    out.sort(key=_latest_key, reverse=True)
    return {"items": out}

def _op_semantic_search_transactions(args: Dict[str, Any], index_dir: str, embedder: Embedder) -> Dict[str, Any]:
    q_raw = (args.get("query") or args.get("category") or "").strip()
    alts = [a for a in (args.get("alternates") or []) if isinstance(a, str) and a.strip()]
    k = int(args.get("k", 5))
    if not q_raw:
        return {"hits": [], "error": "query is required", "trace": {"k": k}}
    merged: Dict[str, Dict[str, Any]] = {}
    for q in [q_raw] + alts[:7]:
        for h in query_index("transactions", q, top_k=max(1, k), index_dir=index_dir, embedder=embedder):
            p = h.get("payload") or {}
            rid = p.get("transactionId") or h.get("idx")
            score = float(h.get("score", 0.0))
            prev = merged.get(rid)
            if (prev is None) or (score > prev["score"]):
                merged[rid] = {"score": score, "text": h.get("text"), "payload": p, "matched_query": q}
    hits = list(merged.values())
    hits.sort(key=lambda x: x["score"], reverse=True)
    return {"hits": hits[:k], "trace": {"k": k, "query": q_raw, "alternates": alts}}

# ----------------------------- main executor ----------------------------------
def execute_calls(calls: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg expects:
      - app_yaml: path to app.yaml (embeddings, index dir)
      - intent: optional string
      - question: original user text (for RAG lane)
      - session_id: conversation id for memory
    """
    results: Dict[str, Any] = {}
    calls = _normalize_calls(calls)
    # config for embeddings / FAISS
    app_yaml = (cfg or {}).get("app_yaml") or "config/app.yaml"
    app_cfg = _read_app_cfg(app_yaml)
    index_dir = ((app_cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    embedder = _build_embedder_from_cfg(app_cfg)

    # handy values for RAG
    session_id = (cfg or {}).get("session_id") or "default"
    question   = (cfg or {}).get("question") or ""

    # we’ll lazy-load datasets per account_id when needed
    cache: Dict[str, Dict[str, Any]] = {}  # account_id -> {domain->data}

    def _ensure_datasets(aid: str) -> Dict[str, Any]:
        ds = cache.get(aid)
        if ds is None:
            ds = {
                "transactions": load_transactions(aid),
                "payments":     load_payments(aid),
                "statements":   load_statements(aid),
                "account_summary": load_account_summary(aid),
            }
            cache[aid] = ds
        return ds

    for i, call in enumerate(calls or []):
        dom = str(call.get("domain_id", "")).strip().lower().replace("-", "_")
        cap = str(call.get("capability", "")).strip().lower().replace("-", "_")
        args = dict(call.get("args") or {})
        key  = f"{dom}.{cap}[{i}]"

        # pick account_id when relevant
        account_id = args.get("account_id") or (cfg or {}).get("account_id")

        # ---------- RAG lane ONLY when domain_id == "rag" ----------
        if dom == "rag":
            k = int(args.get("k", 6))
            try:
                if cap in {"unified_answer", "unified"}:
                    if not account_id:
                        res = {"error": "account_id is required for rag.unified_answer"}
                    else:
                        res = unified_rag_answer(
                            question=question,
                            session_id=session_id,
                            account_id=account_id,
                            k=k,
                        )
                elif cap in {"account_answer", "account_only"}:
                    if not account_id:
                        res = {"error": "account_id is required for rag.account_answer"}
                    else:
                        res = account_rag_answer(
                            question=question,
                            session_id=session_id,
                            account_id=account_id,
                            k=k,
                        )
                elif cap in {"knowledge_answer", "policy_answer", "handbook_answer"}:
                    res = knowledge_rag_answer(
                        question=question,
                        session_id=session_id,
                        k=k,
                    )
                else:
                    res = {"error": f"Unknown rag capability: {cap}"}
            except FileNotFoundError as e:
                res = {"error": f"Index not found: {e}"}
            except Exception as e:
                res = {"error": f"RAG execution error: {e}"}

            results[key] = res
            continue  # do not fall through

        # ---------- Generic DSL lane (NO legacy, NO RAG fallback) ----------
        if cap in {"get_field", "find_latest", "sum_where", "topk_by_sum", "list_where", "semantic_search"}:
            ds = _ensure_datasets(account_id)  # your existing loader bundle
            try:
                if cap == "get_field":
                    res = _op_get_field(dom, args, ds)
                elif cap == "find_latest":
                    # (optional) you can add a tiny generic impl if you need it
                    res = {"error": "find_latest not implemented"}
                elif cap == "sum_where":
                    res = {"error": "sum_where not implemented"}
                elif cap == "topk_by_sum":
                    res = {"error": "topk_by_sum not implemented"}
                elif cap == "list_where":
                    res = {"error": "list_where not implemented"}
                elif cap == "semantic_search":
                    # keep your FAISS semantic search here if you already had it
                    res = {"error": "semantic_search not implemented"}
                else:
                    res = {"error": f"Unknown DSL capability: {cap}"}
            except Exception as e:
                res = {"error": f"DSL execution error: {e}"}

            results[key] = res
            continue

        # ---------- Unknown domain/op (explicit) ----------
        results[key] = {"error": f"Unknown capability {cap} for domain {dom}"}

    return results