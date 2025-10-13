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
def _op_get_field(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Works across ANY domain.
    Args:
      field: path or leaf key
      where: optional dict of equality filters (applied when dataset is a list)
      latest: bool – pick the newest row by timestamp fields
    """
    field = (args.get("field") or "").strip()
    latest = bool(args.get("latest"))
    where  = args.get("where") or {}

    data = datasets.get(domain)
    if data is None:
        return {"error": f"Unknown domain {domain}"}

    if isinstance(data, dict):  # account_summary style
        return {"value": _get_field_from_row(data, field)}

    rows: List[Dict[str, Any]] = data or []
    cand = []
    for r in rows:
        ok = True
        for k, v in where.items():
            if (r.get(k) != v):
                ok = False; break
        if ok:
            cand.append(r)
    if not cand:
        cand = rows
    if not cand:
        return {"value": None}
    row = max(cand, key=_latest_key) if latest else cand[0]
    return {"value": _get_field_from_row(row, field), "row": row}

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
        account_id = args.get("account_id") or (cfg or {}).get("account_id")  # optional

        # --------- RAG lane ---------------------------------------------------
        if dom == "rag":
            try:
                if cap in {"unified_answer", "account_answer"}:
                    ds = _ensure_datasets(account_id) if account_id else {
                        "transactions": [], "payments": [], "statements": [], "account_summary": {}
                    }
                    knowledge_paths = [
                        "data/knowledge/handbook.md",
                        "data/agreement/Apple-Card-Customer-Agreement.pdf",
                    ]
                    res = unified_rag_answer(
                        question=question or args.get("question", ""),
                        session_id=session_id,
                        account_id=account_id or "default",
                        txns=ds["transactions"], pays=ds["payments"],
                        stmts=ds["statements"],  acct=ds["account_summary"],
                        knowledge_paths=knowledge_paths,
                        top_k=int(args.get("k", 5))
                    )
                elif cap == "knowledge_answer":
                    res = knowledge_rag_answer(
                        question=question or args.get("question", ""),
                        session_id=session_id,
                        k=int(args.get("k", 5))
                    )
                else:
                    res = {"error": f"Unknown rag capability '{cap}'"}
            except Exception as e:
                res = {"error": f"RAG failure: {e}"}
            results[key] = res
            continue

        # --------- Generic 5-op DSL (works for any domain) --------------------
        if cap in {"get_field", "find_latest", "sum_where", "topk_by_sum", "list_where", "semantic_search"}:
            try:
                ds = _ensure_datasets(account_id) if account_id else {
                    "transactions": [], "payments": [], "statements": [], "account_summary": {}
                }
                if cap == "get_field":
                    res = _op_get_field(dom, args, ds)
                elif cap == "find_latest":
                    res = _op_find_latest(dom, args, ds)
                elif cap == "sum_where":
                    res = _op_sum_where(dom, args, ds)
                elif cap == "topk_by_sum":
                    res = _op_topk_by_sum(dom, args, ds)
                elif cap == "list_where":
                    res = _op_list_where(dom, args, ds)
                else:  # semantic_search
                    if dom != "transactions":
                        res = {"error": "semantic_search currently supported for transactions only"}
                    else:
                        res = _op_semantic_search_transactions(args, index_dir=index_dir, embedder=embedder)
            except Exception as e:
                res = {"error": f"{cap} failed: {e}"}
            results[key] = res
            continue

        # --------- Legacy deterministic calculators ---------------------------
        try:
            if dom == "transactions":
                ds = _ensure_datasets(account_id)
                txns = ds["transactions"]
                if cap == "last_transaction":
                    cand = [t for t in txns if (not account_id or t.get("accountId") == account_id)]
                    last = max(cand, key=_latest_key) if cand else None
                    res = {"item": last, "trace": {"count": len(cand)}}
                elif cap == "top_merchants":
                    period = args.get("period")
                    rows = [t for t in txns if _within_period(t, period)]
                    rows = [t for t in rows if (t.get("transactionType") == "DEBIT" or (t.get("amount") or 0) > 0)]
                    agg: Dict[str, float] = {}
                    for t in rows:
                        m = (t.get("merchantName") or "UNKNOWN").strip()
                        agg[m] = agg.get(m, 0.0) + float(t.get("amount") or 0.0)
                    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
                    res = {"top_merchants": [{"merchant": m, "total": v} for m, v in top],
                           "trace": {"count": len(rows), "period": period or "ALL"}}
                elif cap == "spend_in_period":
                    res = txn_calc.spend_in_period(txns, account_id, args.get("period"))
                elif cap == "list_over_threshold":
                    thr = float(args.get("threshold", 0))
                    res = txn_calc.list_over_threshold(txns, account_id, thr, args.get("period"))
                elif cap == "average_per_month":
                    res = txn_calc.average_per_month(txns, account_id, args.get("period"), None, None, False)
                elif cap == "find_by_merchant":
                    q = (args.get("merchant_query") or "").strip().lower()
                    hits = [t for t in txns if q and q in (t.get("merchantName") or "").lower()]
                    hits.sort(key=_latest_key, reverse=True)
                    res = {"items": hits, "count": len(hits)}
                else:
                    res = {"error": f"Unknown capability {cap}"}

            elif dom == "payments":
                ds = _ensure_datasets(account_id)
                pays = ds["payments"]
                if cap == "last_payment":
                    res = pay_calc.last_payment(pays, account_id)
                elif cap == "total_credited_year":
                    yr = args.get("year")
                    res = pay_calc.total_credited_year(pays, account_id, int(yr) if yr else None)
                elif cap == "payments_in_period":
                    res = pay_calc.payments_in_period(pays, account_id, args.get("period"))
                else:
                    res = {"error": f"Unknown capability {cap}"}

            elif dom == "statements":
                ds = _ensure_datasets(account_id)
                stmts = ds["statements"]
                if cap == "total_interest":
                    prd = args.get("period")
                    if prd is None:
                        # pick latest, optionally nonzero
                        nz = bool(args.get("nonzero"))
                        cand = [s for s in stmts if (s.get("interestCharged") or 0) > 0] if nz else stmts
                        if not cand:
                            res = {"error": "No statements"}
                        else:
                            row = max(cand, key=_latest_key)
                            res = {"interest_total": row.get("interestCharged") or 0.0,
                                   "trace": {"period": row.get("period")}}
                    else:
                        row = next((s for s in stmts if s.get("period") == prd), None)
                        res = {"interest_total": (row or {}).get("interestCharged") or 0.0,
                               "trace": {"period": prd}}
                elif cap == "interest_breakdown":
                    res = stmt_calc.interest_breakdown(stmts, account_id, args.get("period"))
                elif cap == "trailing_interest":
                    res = stmt_calc.trailing_interest(stmts, account_id, args.get("period"))
                else:
                    res = {"error": f"Unknown capability {cap}"}

            elif dom == "account_summary":
                ds = _ensure_datasets(account_id)
                acct = ds["account_summary"]
                if cap == "current_balance":
                    res = acct_calc.current_balance(acct)
                elif cap == "available_credit":
                    res = acct_calc.available_credit(acct)
                elif cap == "get_field":
                    res = _op_get_field("account_summary", args, ds)
                else:
                    res = {"error": f"Unknown capability {cap}"}

            else:
                # optional: route unknown domains to RAG fallback
                ds = _ensure_datasets(account_id) if account_id else {
                    "transactions": [], "payments": [], "statements": [], "account_summary": {}
                }
                res = unified_rag_answer(
                    question=question,
                    session_id=session_id,
                    account_id=account_id or "default",
                    txns=ds["transactions"], pays=ds["payments"],
                    stmts=ds["statements"],  acct=ds["account_summary"],
                    knowledge_paths=[
                        "data/knowledge/handbook.md",
                        "data/agreement/Apple-Card-Customer-Agreement.pdf",
                    ],
                    top_k=5
                )
        except Exception as e:
            res = {"error": f"Execution failed for {dom}.{cap}: {e}"}

        results[key] = res

    return results