# core/orchestrator/execute.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import os

# ---- domain loaders (your existing modules) ---------------------------------
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

# ---- existing calculators (keep using yours) --------------------------------
from domains.transactions import calculator as txn_calc
from domains.payments import calculator as pay_calc
from domains.statements import calculator as stmt_calc
from domains.account_summary import calculator as acct_calc

# ---- optional semantic search (present in some branches) --------------------
try:
    from core.index.faiss_registry import query_index, Embedder  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# =============================================================================
# Helpers
# =============================================================================

def _as_str(x: Any) -> str:
    return "" if x is None else str(x)

def _to_dt(s: Optional[str]) -> Optional[datetime]:
    """ISO 8601 tolerant parser (handles 'Z')."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _txn_ts(t: Dict[str, Any]) -> str:
    """Best-effort timestamp field preference for transactions."""
    return (
        t.get("transactionDateTime")
        or t.get("postedDateTime")
        or t.get("date")
        or ""
    )

def _within_period(txn: Dict[str, Any], period: Optional[str]) -> bool:
    """
    Supported period examples:
      - None / ""        -> no filter
      - "LAST_12M"       -> last 365 days
      - "YYYY-MM"        -> exact month match
    """
    if not period:
        return True

    p = period.strip().upper()
    ts = _to_dt(_txn_ts(txn))
    if not ts:
        return False

    if p == "LAST_12M":
        return ts >= (datetime.now(timezone.utc) - timedelta(days=365))

    # YYYY-MM
    if len(period) == 7 and period[4] == "-":
        return ts.strftime("%Y-%m") == period

    # Fallback: no filter if an unknown format is given
    return True

def _pick_account_id(calls: List[dict], cfg: Dict[str, Any]) -> Optional[str]:
    """Prefer explicit config_paths.account_id; else first call arg.account_id if present."""
    aid = (cfg or {}).get("account_id")
    if aid:
        return str(aid)
    for c in calls or []:
        a = c.get("args") or {}
        if a.get("account_id"):
            return str(a["account_id"])
    return None

def _fmt_money(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

# Generic field aliases to avoid brittle names in account_summary
FIELD_ALIASES: Dict[str, List[str]] = {
    "accountStatus": ["status", "account status", "state", "account_state"],
    "currentBalance": ["current balance", "balance", "statementBalance", "currentAdjustedBalance"],
    "availableCreditAmount": ["availableCredit", "available credit", "available credit amount"],
    "creditLimit": ["credit limit"],
    "minimumDueAmount": ["minimum due", "minimum payment due"],
    "paymentDueDate": ["due date", "payment due date", "next due date"],
    "accountNumberLast4": ["last 4", "last4"],
}

def _get_field_with_alias(obj: Dict[str, Any], key: str) -> Any:
    """Return value at canonical key or any known alias (case-insensitive)."""
    if key in obj:
        return obj[key]
    low = key.lower()
    if low in obj:
        return obj[low]
    # try aliases
    for canonical, alts in FIELD_ALIASES.items():
        if canonical == key:
            # try canonical via different casing
            if canonical in obj:
                return obj[canonical]
            if canonical.lower() in obj:
                return obj[canonical.lower()]
        for alt in alts:
            if alt in obj:
                return obj[alt]
            if alt.lower() in obj:
                return obj[alt.lower()]
    return None

# =============================================================================
# Main executor
# =============================================================================

def execute_calls(calls: List[dict], config_paths: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic executor for planner-produced calls.

    Inputs:
      - calls: [{domain_id, capability, args}]
      - config_paths: {app_yaml?, intent?, account_id?}

    Behavior:
      - Loads 4 JSON files for the chosen account_id
      - Runs domain-specific calculators OR generic get_field
      - Enforces "POSTED + PURCHASE + positive amount" for merchant spend
      - Provides optional 'transactions.semantic_search' if FAISS is present
    """
    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 0) Resolve account & load data
    # -------------------------------------------------------------------------
    account_id = _pick_account_id(calls, config_paths) or "default"

    txns = load_transactions(account_id)
    pays = load_payments(account_id)
    stmts = load_statements(account_id)
    acct  = load_account_summary(account_id)

    # latest statement period (if you need a default)
    stmt_periods = [s.get("period") for s in stmts if isinstance(s, dict) and s.get("period")]
    latest_stmt_period = max(stmt_periods) if stmt_periods else None

    # -------------------------------------------------------------------------
    # 1) Execute each planned call
    # -------------------------------------------------------------------------
    for i, call in enumerate(calls or []):
        dom = _as_str(call.get("domain_id")).strip().lower().replace("-", "_")
        cap = _as_str(call.get("capability")).strip().lower().replace("-", "_")
        args = dict(call.get("args") or {})
        key  = f"{account_id}:{dom}.{cap}[{i}]"

        res: Any = {"error": f"Unknown domain {dom}"}

        # ======================= TRANSACTIONS =================================
        if dom == "transactions":
            if cap == "last_transaction":
                aid = args.get("account_id")
                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                if not cand:
                    res = {"error": "No transactions", "trace": {"count": 0}}
                else:
                    last = max(cand, key=_txn_ts)
                    res = {"item": last}

            elif cap == "top_merchants":
                # posted + purchase + positive only
                period = args.get("period")
                aid    = args.get("account_id")
                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                cand = [t for t in cand if _within_period(t, period)]
                cand = [
                    t for t in cand
                    if _as_str(t.get("transactionStatus")) == "POSTED"
                    and "PURCHASE" in _as_str(t.get("displayTransactionType")).upper()
                    and _fmt_money(t.get("amount")) > 0
                ]
                totals: Dict[str, float] = {}
                for t in cand:
                    m = (t.get("merchantName") or "UNKNOWN").strip()
                    totals[m] = totals.get(m, 0.0) + _fmt_money(t.get("amount"))
                top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
                res = {
                    "top_merchants": [{"merchant": m, "total": v} for m, v in top],
                    "trace": {"count": len(cand), "period": period or "ALL"}
                }

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
                    res = {"items": hits, "count": len(hits),
                           "trace": {"merchant_query": q, "period": period or "ALL"}}

            elif cap == "list_over_threshold":
                thr = float(args.get("threshold", 0))
                res = txn_calc.list_over_threshold(txns, args.get("account_id"), thr, args.get("period"))

            elif cap == "spend_in_period":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))

            elif cap == "purchases_in_cycle":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))

            elif cap == "max_amount":
                res = txn_calc.max_amount(
                    txns,
                    args.get("account_id"),
                    args.get("period"),
                    args.get("period_start"),
                    args.get("period_end"),
                    args.get("category"),
                    int(args.get("top", 1)),
                )

            elif cap == "aggregate_by_category":
                res = txn_calc.aggregate_by_category(
                    txns,
                    args.get("account_id"),
                    args.get("period"),
                    args.get("period_start"),
                    args.get("period_end"),
                )

            elif cap == "average_per_month":
                res = txn_calc.average_per_month(
                    txns,
                    args.get("account_id"),
                    args.get("period"),
                    args.get("period_start"),
                    args.get("period_end"),
                    bool(args.get("include_credits", False)),
                )

            elif cap == "semantic_search":
                # Optional: only if FAISS available; otherwise return a clear error
                if not _FAISS_AVAILABLE:
                    res = {"error": "semantic_search not available (FAISS registry missing)"}
                else:
                    q_raw = _as_str(args.get("query") or args.get("category")).strip()
                    k     = int(args.get("k", 5))
                    if not q_raw:
                        res = {"hits": [], "error": "query is required", "trace": {"k": k}}
                    else:
                        # You already have index building; we assume index_dir/env are configured elsewhere.
                        # If your query_index signature differs, adapt here.
                        hits = query_index("transactions", q_raw, top_k=max(1, k))
                        res  = {"hits": hits, "trace": {"k": k, "query": q_raw}}

            else:
                res = {"error": f"Unknown capability {cap}"}

        # ======================= PAYMENTS =====================================
        elif dom == "payments":
            if cap == "last_payment":
                res = pay_calc.last_payment(pays, args.get("account_id"))
            elif cap == "payments_in_period":
                res = pay_calc.payments_in_period(pays, args.get("account_id"), args.get("period"))
            elif cap == "total_credited_year":
                yr = args.get("year")
                res = pay_calc.total_credited_year(pays, args.get("account_id"), int(yr) if yr else None)
            else:
                res = {"error": f"Unknown capability {cap}"}

        # ======================= STATEMENTS ===================================
        elif dom == "statements":
            # default the period (latest, or latest non-zero if requested)
            if not args.get("period"):
                if args.get("nonzero"):
                    nz = [s.get("period") for s in stmts if _fmt_money(s.get("interestCharged")) > 0]
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
                        "interest_total": _fmt_money(row.get("interestCharged")),
                        "trace": {"period": prd, "close_date": row.get("closingDateTime")}
                    }

            elif cap == "interest_breakdown":
                res = stmt_calc.interest_breakdown(stmts, args.get("account_id"), args.get("period"))

            elif cap == "trailing_interest":
                res = stmt_calc.trailing_interest(stmts, args.get("account_id"), args.get("period"))

            else:
                res = {"error": f"Unknown capability {cap}"}

        # ======================= ACCOUNT SUMMARY ==============================
        elif dom == "account_summary":
            if cap == "current_balance":
                res = acct_calc.current_balance(acct)

            elif cap == "available_credit":
                res = acct_calc.available_credit(acct)

            elif cap == "get_field":
                # Generic: support literal field Qs without writing new code
                # args: { key_path | key }
                key = _as_str(args.get("key_path") or args.get("key")).strip()
                if not key:
                    res = {"error": "key_path is required for get_field"}
                else:
                    val = _get_field_with_alias(acct, key)
                    res = {"value": val}

            else:
                res = {"error": f"Unknown capability {cap}"}

        # ======================= POLICY (optional) ============================
        elif dom == "policy":
            res = {"error": "policy domain not wired in executor"}

        # -------------------- store result ------------------------------------
        results[key] = res

    return results