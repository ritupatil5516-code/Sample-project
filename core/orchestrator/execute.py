# core/orchestrator/execute.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
import json
import os

# ---- domain loaders
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

# ---- calculators (keep)
from domains.transactions import calculator as txn_calc
from domains.payments import calculator as pay_calc
from domains.statements import calculator as stmt_calc
from domains.account_summary import calculator as acct_calc

# ---- optional semantic search (if present)
try:
    from core.index.faiss_registry import query_index  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# ---- RAG (LangChain conversational retrieval)
from core.retrieval.rag_chain import account_rag_answer, knowledge_rag_answer

# =============================================================================
# Helpers
# =============================================================================

def _as_str(x: Any) -> str:
    return "" if x is None else str(x)

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _txn_ts(t: Dict[str, Any]) -> str:
    return (
        t.get("transactionDateTime")
        or t.get("postedDateTime")
        or t.get("date")
        or ""
    )

def _within_period(txn: Dict[str, Any], period: Optional[str]) -> bool:
    if not period:
        return True
    p = period.strip().upper()
    ts = _to_dt(_txn_ts(txn))
    if not ts:
        return False
    if p == "LAST_12M":
        return ts >= (datetime.now(timezone.utc) - timedelta(days=365))
    if len(period) == 7 and period[4] == "-":  # YYYY-MM
        return ts.strftime("%Y-%m") == period
    return True

def _pick_account_id(calls: List[dict], cfg: Dict[str, Any]) -> Optional[str]:
    aid = (cfg or {}).get("account_id")
    if aid:
        return str(aid)
    for c in calls or []:
        a = c.get("args") or {}
        if a.get("account_id"):
            return str(a["account_id"])
    return None

def _get_nested(d: Any, dotted: str) -> Any:
    cur = d
    for part in (dotted or "").split("."):
        if part == "":
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur

def _apply_filters(rows: List[Dict[str, Any]], flt: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not flt:
        return rows
    out = []
    for r in rows:
        ok = True
        for k, want in flt.items():
            contains = False
            if isinstance(k, str) and k.endswith("~"):
                contains = True
                k = k[:-1]
            got = _get_nested(r, k)
            if contains:
                ok = (_as_str(want).lower() in _as_str(got).lower())
            else:
                ok = (_as_str(got) == _as_str(want))
            if not ok:
                break
        if ok:
            out.append(r)
    return out

def _sort_rows(rows: List[Dict[str, Any]], sort_by: Optional[str], order: str) -> List[Dict[str, Any]]:
    if not sort_by:
        return rows
    rev = (order or "desc").lower().startswith("d")
    return sorted(rows, key=lambda r: (_get_nested(r, sort_by) is None, _get_nested(r, sort_by)), reverse=rev)

def _project_values(rows: List[Dict[str, Any]], key_path: str) -> List[Any]:
    return [_get_nested(r, key_path) for r in rows]

def _aggregate(values: List[Any], agg: Optional[str]) -> Any:
    if not agg:
        return values
    a = (agg or "").lower()
    if not values:
        return None if a in ("first", "last", "max", "min", "avg", "sum") else 0
    if a == "first": return values[0]
    if a == "last":  return values[-1]
    if a == "max":
        try: return max(values)
        except Exception: return None
    if a == "min":
        try: return min(values)
        except Exception: return None
    if a == "sum": return sum(_to_float(v, 0.0) for v in values)
    if a == "avg":
        nums = [_to_float(v, 0.0) for v in values]
        return (sum(nums) / len(nums)) if nums else 0.0
    if a == "count": return len(values)
    if a == "unique":
        try:
            # JSON-string unique; lightweight
            return list({json.dumps(v, sort_keys=True) for v in values})
        except Exception:
            return list({str(v) for v in values})
    return values

FIELD_ALIASES: Dict[str, List[str]] = {
    "accountStatus": ["status", "account status", "state", "account_state"],
    "currentBalance": ["current balance", "balance", "statementBalance", "currentAdjustedBalance"],
    "availableCreditAmount": ["available credit", "availableCredit", "available credit amount"],
    "creditLimit": ["credit limit"],
    "minimumDueAmount": ["minimum due", "minimum payment due"],
    "paymentDueDate": ["due date", "payment due date", "next due date"],
    "accountNumberLast4": ["last 4", "last4", "account last 4"],
}

def _get_field_with_alias(obj: Dict[str, Any], key: str) -> Any:
    if key in obj: return obj[key]
    low = key.lower()
    if low in obj: return obj[low]
    for canonical, alts in FIELD_ALIASES.items():
        if canonical == key:
            if canonical in obj: return obj[canonical]
            if canonical.lower() in obj: return obj[canonical.lower()]
        for alt in alts:
            if alt in obj: return obj[alt]
            if alt.lower() in obj: return obj[alt.lower()]
    got = _get_nested(obj, key)
    return got

# =============================================================================
# Executor
# =============================================================================

def execute_calls(calls: List[dict], config_paths: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes planner calls with:
      - universal get_field across all domains
      - deterministic calculators (legacy)
      - optional semantic_search
      - RAG (LangChain conversational memory)
    """
    results: Dict[str, Any] = {}

    account_id = (config_paths or {}).get("account_id") or _pick_account_id(calls, config_paths) or "default"
    original_question = (config_paths or {}).get("question")
    session_id = (config_paths or {}).get("session_id") or "default"

    # Load per-account data
    txns = load_transactions(account_id)
    pays = load_payments(account_id)
    stmts = load_statements(account_id)
    acct  = load_account_summary(account_id)

    stmt_periods = [s.get("period") for s in stmts if isinstance(s, dict) and s.get("period")]
    latest_stmt_period = max(stmt_periods) if stmt_periods else None

    for i, call in enumerate(calls or []):
        dom = _as_str(call.get("domain_id")).strip().lower().replace("-", "_")
        cap = _as_str(call.get("capability")).strip().lower().replace("-", "_")
        args = dict(call.get("args") or {})
        key  = f"{account_id}:{dom}.{cap}[{i}]"
        res: Any = {"error": f"Unknown domain {dom}"}

        # ======================= TRANSACTIONS =================================
        if dom == "transactions":
            if cap == "get_field":
                key_path = _as_str(args.get("key_path") or args.get("key")).strip()
                if not key_path:
                    res = {"error": "key_path is required"}
                else:
                    rows = txns
                    rows = [t for t in rows if _within_period(t, args.get("period"))]
                    rows = _apply_filters(rows, args.get("filter") or {})
                    rows = _sort_rows(rows, args.get("sort_by"), _as_str(args.get("order") or "desc"))
                    if args.get("limit"):
                        try: rows = rows[: int(args["limit"])]
                        except Exception: pass
                    values = _project_values(rows, key_path)
                    agg_val = _aggregate(values, args.get("agg"))
                    res = {
                        "values": agg_val if args.get("agg") in (None, "unique") else ([agg_val] if args.get("agg") else values),
                        "count": len(rows),
                        "trace": {"key_path": key_path, "filtered": len(rows)}
                    }

            elif cap == "last_transaction":
                aid = args.get("account_id")
                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                if not cand:
                    res = {"error": "No transactions", "trace": {"count": 0}}
                else:
                    last = max(cand, key=_txn_ts)
                    res = {"item": last}

            elif cap == "top_merchants":
                period = args.get("period")
                aid    = args.get("account_id")
                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                cand = [t for t in cand if _within_period(t, period)]
                cand = [
                    t for t in cand
                    if _as_str(t.get("transactionStatus")).upper() == "POSTED"
                    and "PURCHASE" in _as_str(t.get("displayTransactionType")).upper()
                    and _to_float(t.get("amount"), 0.0) > 0
                ]
                totals: Dict[str, float] = {}
                for t in cand:
                    m = (t.get("merchantName") or "UNKNOWN").strip()
                    totals[m] = totals.get(m, 0.0) + _to_float(t.get("amount"), 0.0)
                top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
                res = {"top_merchants": [{"merchant": m, "total": v} for m, v in top],
                       "trace": {"count": len(cand), "period": period or "ALL"}}

            elif cap == "find_by_merchant":
                q = _as_str(args.get("merchant_query")).strip().lower()
                if not q:
                    res = {"error": "merchant_query is required"}
                else:
                    period = args.get("period")
                    aid    = args.get("account_id")
                    cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                    cand = [t for t in cand if _within_period(t, period)]
                    hits = [t for t in cand if q in _as_str(t.get("merchantName")).lower()]
                    hits.sort(key=_txn_ts, reverse=True)
                    res = {"items": hits, "count": len(hits), "trace": {"merchant_query": q, "period": period or "ALL"}}

            elif cap == "list_over_threshold":
                thr = float(args.get("threshold", 0))
                res = txn_calc.list_over_threshold(txns, args.get("account_id"), thr, args.get("period"))

            elif cap == "spend_in_period":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))

            elif cap == "purchases_in_cycle":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))

            elif cap == "max_amount":
                res = txn_calc.max_amount(
                    txns, args.get("account_id"), args.get("period"),
                    args.get("period_start"), args.get("period_end"),
                    args.get("category"), int(args.get("top", 1)),
                )

            elif cap == "aggregate_by_category":
                res = txn_calc.aggregate_by_category(
                    txns, args.get("account_id"),
                    args.get("period"), args.get("period_start"), args.get("period_end"),
                )

            elif cap == "average_per_month":
                res = txn_calc.average_per_month(
                    txns, args.get("account_id"),
                    args.get("period"), args.get("period_start"), args.get("period_end"),
                    bool(args.get("include_credits", False)),
                )

            elif cap == "semantic_search":
                if not _FAISS_AVAILABLE:
                    res = {"error": "semantic_search not available (FAISS missing)"}
                else:
                    q_raw = _as_str(args.get("query") or args.get("category")).strip()
                    k     = int(args.get("k", 5))
                    if not q_raw:
                        res = {"hits": [], "error": "query is required", "trace": {"k": k}}
                    else:
                        hits = query_index("transactions", q_raw, top_k=max(1, k))
                        res  = {"hits": hits, "trace": {"k": k, "query": q_raw}}
            else:
                res = {"error": f"Unknown capability {cap}"}

        # ============================ PAYMENTS ================================
        elif dom == "payments":
            if cap == "get_field":
                key_path = _as_str(args.get("key_path") or args.get("key")).strip()
                if not key_path:
                    res = {"error": "key_path is required"}
                else:
                    rows = _apply_filters(pays, args.get("filter") or {})
                    rows = _sort_rows(rows, args.get("sort_by"), _as_str(args.get("order") or "desc"))
                    if args.get("limit"):
                        try: rows = rows[: int(args["limit"])]
                        except Exception: pass
                    values = _project_values(rows, key_path)
                    agg_val = _aggregate(values, args.get("agg"))
                    res = {
                        "values": agg_val if args.get("agg") in (None, "unique") else ([agg_val] if args.get("agg") else values),
                        "count": len(rows),
                        "trace": {"key_path": key_path, "filtered": len(rows)}
                    }

            elif cap == "last_payment":
                res = pay_calc.last_payment(pays, args.get("account_id"))
            elif cap == "payments_in_period":
                res = pay_calc.payments_in_period(pays, args.get("account_id"), args.get("period"))
            elif cap == "total_credited_year":
                yr = args.get("year")
                res = pay_calc.total_credited_year(pays, args.get("account_id"), int(yr) if yr else None)
            else:
                res = {"error": f"Unknown capability {cap}"}

        # ============================ STATEMENTS ==============================
        elif dom == "statements":
            if cap == "get_field":
                key_path = _as_str(args.get("key_path") or args.get("key")).strip()
                if not key_path:
                    res = {"error": "key_path is required"}
                else:
                    rows = _apply_filters(stmts, args.get("filter") or {})
                    rows = _sort_rows(rows, args.get("sort_by"), _as_str(args.get("order") or "desc"))
                    if args.get("limit"):
                        try: rows = rows[: int(args["limit"])]
                        except Exception: pass
                    values = _project_values(rows, key_path)
                    agg_val = _aggregate(values, args.get("agg"))
                    res = {
                        "values": agg_val if args.get("agg") in (None, "unique") else ([agg_val] if args.get("agg") else values),
                        "count": len(rows),
                        "trace": {"key_path": key_path, "filtered": len(rows)}
                    }

            else:
                if not args.get("period"):
                    if args.get("nonzero"):
                        nz = [s.get("period") for s in stmts if _to_float(s.get("interestCharged")) > 0]
                        args["period"] = max(nz) if nz else latest_stmt_period
                    else:
                        args["period"] = latest_stmt_period

                if cap == "total_interest":
                    prd = args.get("period")
                    row = next((s for s in stmts if s.get("period") == prd), None)
                    if not row:
                        res = {"error": "No statement for period", "trace": {"period": prd}}
                    else:
                        res = {
                            "interest_total": _to_float(row.get("interestCharged")),
                            "trace": {"period": prd, "close_date": row.get("closingDateTime")}
                        }
                elif cap == "interest_breakdown":
                    res = stmt_calc.interest_breakdown(stmts, args.get("account_id"), args.get("period"))
                elif cap == "trailing_interest":
                    res = stmt_calc.trailing_interest(stmts, args.get("account_id"), args.get("period"))
                else:
                    res = {"error": f"Unknown capability {cap}"}

        # ======================== ACCOUNT SUMMARY =============================
        elif dom == "account_summary":
            if cap == "get_field":
                key_path = _as_str(args.get("key_path") or args.get("key")).strip()
                if not key_path:
                    res = {"error": "key_path is required"}
                else:
                    val = _get_field_with_alias(acct, key_path)
                    if val is None:
                        val = _get_nested(acct, key_path)
                    res = {"value": val, "trace": {"key_path": key_path}}

            elif cap == "current_balance":
                res = acct_calc.current_balance(acct)
            elif cap == "available_credit":
                res = acct_calc.available_credit(acct)
            else:
                res = {"error": f"Unknown capability {cap}"}

        # ================================ RAG =================================
        elif dom == "rag":
            scope = (args.get("scope") or "account").lower()
            q = args.get("question") or original_question
            sess = args.get("session_id") or session_id
            if not q:
                res = {"error": "question is required for rag"}
            else:
                if scope == "knowledge":
                    res = knowledge_rag_answer(q, sess, k=int(args.get("k", 6)))
                else:
                    aid = args.get("account_id") or account_id
                    res = account_rag_answer(q, sess, aid, k=int(args.get("k", 6)))

        else:
            res = {"error": f"Unknown domain {dom}"}

        results[key] = res

    return results