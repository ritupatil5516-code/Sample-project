# core/orchestrator/execute.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import math
import json

# Shared runtime (one LLM, one embeddings, per-session memory, config)
from src.core.runtime import RUNTIME

# Domain loaders (they can pull from local JSON or API based on config)
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

# RAG (Option B: LlamaIndex retriever adapted to LangChain)
from core.retrieval.rag_chain import unified_rag_answer


# --------------------------------------------------------------------------------------
# Domain registry: loader + aliases (friendly field names → canonical schema keys)
# --------------------------------------------------------------------------------------

ALIASES: Dict[str, Dict[str, str]] = {
    "transactions": {
        "date": "postedDateTime",        # generic → canonical
        "status": "transactionStatus",
        "type": "displayTransactionType",
        "merchant": "merchantName",
        "amount": "amount",
        "category": "category",
        # add more as you discover variations
    },
    "payments": {
        "date": "paymentPostedDateTime",
        "amount": "amount",
        "status": "status",
    },
    "statements": {
        "period": "period",
        "close": "closingDateTime",
        "interest": "interestCharged",
        "balance": "statementBalance",
    },
    "accounts": {
        "status": "accountStatus",
        "account_status": "accountStatus",
        "balance": "currentBalance",
        "current_balance": "currentBalance",
        "available": "availableCredit",
        "available_credit": "availableCredit",
        "credit_limit": "creditLimit",
    },
}

# which domains are list-shaped vs dict-shaped
LISTY = {"transactions", "payments", "statements"}
DICTY = {"accounts"}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _alias(domain: str, field: str) -> str:
    if not field:
        return field
    amap = ALIASES.get(domain, {})
    return amap.get(field, field)

def _get_dict_path(row: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Supports dotted paths like "foo.bar.baz". No array indexing to keep it simple.
    """
    cur: Any = row
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

_TS_KEYS = (
    "postedDateTime",
    "transactionDateTime",
    "paymentPostedDateTime",
    "paymentDateTime",
    "closingDateTime",
    "openingDateTime",
    "date",
)

def _row_ts(row: Dict[str, Any]) -> str:
    for k in _TS_KEYS:
        v = row.get(k)
        if v:
            return str(v)
    return ""

def _is_number(x: Any) -> bool:
    try:
        _ = float(x)
        return True
    except Exception:
        return False


# --------------------- filtering (mini-DSL) ---------------------

def _match_cell(val: Any, cond: Any) -> bool:
    """
    cell match logic:
      - dict with ops: {$contains, $in, $eq, $ne, $gte, $lte, $gt, $lt}
      - '*' suffix wildcard for string equals/startswith
      - bare str/number → equality
    """
    # operator object
    if isinstance(cond, dict):
        # contains (case-insensitive for strings)
        if "$contains" in cond:
            needle = str(cond["$contains"]).lower()
            return needle in str(val or "").lower()
        # in (list of accepted values)
        if "$in" in cond:
            arr = cond["$in"] or []
            return str(val) in [str(a) for a in arr]
        # equality
        if "$eq" in cond:
            return str(val) == str(cond["$eq"])
        if "$ne" in cond:
            return str(val) != str(cond["$ne"])
        # numeric comparisons (best effort)
        try:
            fv = float(val)
        except Exception:
            fv = None
        if fv is not None:
            if "$gte" in cond and not (fv >= float(cond["$gte"])): return False
            if "$lte" in cond and not (fv <= float(cond["$lte"])): return False
            if "$gt"  in cond and not (fv >  float(cond["$gt"])):  return False
            if "$lt"  in cond and not (fv <  float(cond["$lt"])):  return False
            return True
        return False

    # wildcard suffix
    if isinstance(cond, str) and cond.endswith("*"):
        return str(val or "").lower().startswith(cond[:-1].lower())

    # bare value → equality (stringified)
    return str(val) == str(cond)

def _match_row(domain: str, row: Dict[str, Any], where: Dict[str, Any]) -> bool:
    if not where:
        return True
    for k, want in (where or {}).items():
        key = _alias(domain, k)
        got = _get_dict_path(row, key, None)
        if not _match_cell(got, want):
            return False
    return True


# --------------------------------------------------------------------------------------
# DSL operations (deterministic, no LLM)
# --------------------------------------------------------------------------------------

def _op_get_field(domain: str, data: Any, args: Dict[str, Any], scratch: Dict[str, Any]) -> Dict[str, Any]:
    """
    For dict-shaped domain (accounts): return a single field value.
    args: { field: "accountStatus" } (aliases supported)
    """
    field = _alias(domain, (args.get("field") or "").strip())
    if not field:
        return {"error": "field is required"}
    if domain in DICTY and isinstance(data, dict):
        return {"value": _get_dict_path(data, field)}
    # allow fallback to latest row in list-shaped domains
    if domain in LISTY and isinstance(data, list) and data:
        latest = max(data, key=_row_ts)
        return {"value": _get_dict_path(latest, field), "row": latest}
    return {"error": "domain not loaded or shape mismatch"}

def _op_find_latest(domain: str, rows: List[Dict[str, Any]], args: Dict[str, Any], scratch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick latest row by timestamp; return requested field + the row for context.
    args: { field: "...", where: {...} }
    """
    field = _alias(domain, (args.get("field") or "").strip())
    where = args.get("where") or {}
    cand = [r for r in (rows or []) if _match_row(domain, r, where)]
    if not cand:
        return {"value": None, "row": None, "trace": {"count": 0, "where": where}}
    latest = max(cand, key=_row_ts)
    return {"value": _get_dict_path(latest, field), "row": latest, "trace": {"count": len(cand), "where": where}}

def _op_sum_where(domain: str, rows: List[Dict[str, Any]], args: Dict[str, Any], scratch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sum a numeric field across filtered rows.
    args: { sum_field?: "amount" (default), where?: {...} }
    """
    where = args.get("where") or {}
    fld = _alias(domain, (args.get("sum_field") or "amount"))
    cand = [r for r in (rows or []) if _match_row(domain, r, where)]
    total = 0.0
    for r in cand:
        v = r.get(fld)
        if _is_number(v):
            total += float(v)
    return {"total": total, "count": len(cand), "trace": {"sum_field": fld, "where": where}}

def _op_topk_by_sum(domain: str, rows: List[Dict[str, Any]], args: Dict[str, Any], scratch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Group by key_field and sum sum_field, return top K.
    args: { key_field: "merchantName", sum_field?: "amount", where?: {...}, k?: 5 }
    """
    where = args.get("where") or {}
    key_field = _alias(domain, (args.get("key_field") or "merchantName"))
    sum_field = _alias(domain, (args.get("sum_field") or "amount"))
    k = int(args.get("k", 5))
    cand = [r for r in (rows or []) if _match_row(domain, r, where)]
    buckets: Dict[str, float] = {}
    for r in cand:
        key = str(_get_dict_path(r, key_field, "UNKNOWN"))
        v = r.get(sum_field)
        if _is_number(v):
            buckets[key] = buckets.get(key, 0.0) + float(v)
    top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:k]
    # store in scratch for follow-ups like “give dates for each spend”
    scratch["last_top_keys"] = [name for name, _ in top]
    scratch["last_top_key_field"] = key_field
    return {
        "top": [{"key": name, "total": total} for name, total in top],
        "trace": {"where": where, "group_by": key_field, "sum": sum_field, "k": k, "rows": len(cand)},
    }

def _op_list_where(domain: str, rows: List[Dict[str, Any]], args: Dict[str, Any], scratch: Dict[str, Any]) -> Dict[str, Any]:
    """
    List rows matching filter; if none specified and we have a recent top-k context,
    restrict to those keys to answer follow-ups.
    args: {
      where?: {...},
      per_key?: "merchantName",          # optional: latest per key
      limit_per_key?: int,               # optional
      limit?: int                        # overall cap
    }
    """
    where = dict(args.get("where") or {})

    # follow-up convenience: if no filter but we have previous top keys, scope by them
    per_key = _alias(domain, (args.get("per_key") or scratch.get("last_top_key_field") or ""))
    last_keys = scratch.get("last_top_keys") if not where else None
    if last_keys and per_key:
        where[per_key] = {"$in": last_keys}

    cand = [r for r in (rows or []) if _match_row(domain, r, where)]
    # latest-first
    cand.sort(key=_row_ts, reverse=True)

    limit = int(args.get("limit", 100))
    limit_per_key = int(args.get("limit_per_key", 0))
    if per_key and limit_per_key > 0:
        # keep only the latest N per group
        seen: Dict[str, int] = {}
        out: List[Dict[str, Any]] = []
        for r in cand:
            key = str(_get_dict_path(r, per_key, "UNKNOWN"))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= limit_per_key:
                out.append(r)
        cand = out

    return {"items": cand[:limit], "trace": {"where": where, "limit": limit, "per_key": per_key, "limit_per_key": limit_per_key}}


# --------------------------------------------------------------------------------------
# Loader wrapper
# --------------------------------------------------------------------------------------

def _load_domain(domain: str, account_id: Optional[str], cfg: Dict[str, Any]) -> Any:
    d = domain.lower()
    if d == "transactions":
        return load_transactions(account_id, cfg)
    if d == "payments":
        return load_payments(account_id, cfg)
    if d == "statements":
        return load_statements(account_id, cfg)
    if d in ("account_summary", "accounts"):
        return load_account_summary(account_id, cfg)
    return None


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------

def execute_calls(calls: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes planner calls with strategy-aware routing:
      - strategy == "deterministic": run DSL op
      - strategy == "rag": run unified RAG (LlamaIndex retriever → LC conversation)
      - strategy == "auto": try deterministic; if empty/None → RAG fallback
    Returns a dict keyed as "domain.capability[i]" with normalized payloads.
    """
    cfg = RUNTIME.cfg or {}
    session_id = str(context.get("session_id", "default"))
    account_id = context.get("account_id")
    question = context.get("question", "")
    allow_rag_fallback = bool((cfg.get("execution") or {}).get("allow_rag_fallback", True))
    top_k = int(context.get("top_k", 6))

    results: Dict[str, Any] = {}

    # cache per-domain data within this request
    domain_cache: Dict[str, Any] = {}

    for i, call in enumerate(calls or []):
        dom = str(call.get("domain_id", "")).strip().lower().replace("-", "_")
        cap = str(call.get("capability", "")).strip().lower().replace("-", "_")
        args = dict(call.get("args") or {})
        strategy = (call.get("strategy") or "deterministic").lower()
        key = f"{dom}.{cap}[{i}]"

        # RAG as its own “domain” (if your planner emits dom="rag")
        if dom == "rag" or strategy == "rag":
            rag_res = unified_rag_answer(
                question=question,
                session_id=session_id,
                account_id=account_id,
                k=top_k,
            )
            results[key] = rag_res
            continue

        # load domain data (once per domain)
        if dom not in domain_cache:
            domain_cache[dom] = _load_domain(dom, account_id, cfg)
        data = domain_cache[dom]

        # -------- deterministic path --------
        deterministic_result: Optional[Dict[str, Any]] = None
        scratch = RUNTIME.get_scratch(session_id) if hasattr(RUNTIME, "get_scratch") else {}

        try:
            if cap == "get_field":
                deterministic_result = _op_get_field(dom, data, args, scratch)
            elif cap == "find_latest":
                if dom in LISTY and isinstance(data, list):
                    deterministic_result = _op_find_latest(dom, data, args, scratch)
                else:
                    deterministic_result = {"error": "find_latest expects a list domain"}
            elif cap == "sum_where":
                if dom in LISTY and isinstance(data, list):
                    deterministic_result = _op_sum_where(dom, data, args, scratch)
                else:
                    deterministic_result = {"error": "sum_where expects a list domain"}
            elif cap == "topk_by_sum":
                if dom in LISTY and isinstance(data, list):
                    deterministic_result = _op_topk_by_sum(dom, data, args, scratch)
                else:
                    deterministic_result = {"error": "topk_by_sum expects a list domain"}
            elif cap == "list_where":
                if dom in LISTY and isinstance(data, list):
                    deterministic_result = _op_list_where(dom, data, args, scratch)
                else:
                    deterministic_result = {"error": "list_where expects a list domain"}
            else:
                deterministic_result = {"error": f"Unknown capability '{cap}' for domain '{dom}'"}
        except Exception as e:
            deterministic_result = {"error": f"op_error: {type(e).__name__}: {e}"}

        # if explicitly deterministic, or success with data → finalize
        if strategy == "deterministic":
            results[key] = deterministic_result
            continue

        # strategy == "auto": if deterministic has meaningful data, keep it
        need_fallback = False
        if deterministic_result is None:
            need_fallback = True
        else:
            # heuristics: empty lists / None values → fallback allowed
            if "items" in deterministic_result and not deterministic_result.get("items"):
                need_fallback = True
            if "value" in deterministic_result and deterministic_result.get("value") in (None, "", []):
                need_fallback = True
            if "top" in deterministic_result and not deterministic_result.get("top"):
                need_fallback = True

        if strategy == "auto" and allow_rag_fallback and need_fallback:
            rag_res = unified_rag_answer(
                question=question,
                session_id=session_id,
                account_id=account_id,
                k=top_k,
            )
            results[key] = {"deterministic": deterministic_result, "fallback": rag_res}
        else:
            results[key] = deterministic_result

    return results