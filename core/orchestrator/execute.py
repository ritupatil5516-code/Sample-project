from typing import Dict, Any, List

from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary
from domains.transactions import calculator as txn_calc
from domains.payments import calculator as pay_calc
from domains.statements import calculator as stmt_calc
from domains.account_summary import calculator as acct_calc
from core.retrieval.policy_index import get_policy_snippet

def _stmt_key(s: dict) -> str:
    # prefer closingDateTime, else period, else empty
    return (s.get("closingDateTime") or s.get("period") or "")

def _pick_latest_stmt(stmts: list[dict], nonzero: bool) -> dict | None:
    if nonzero:
        cand = [s for s in stmts if (s.get("interestCharged") or 0) > 0]
    else:
        cand = stmts
    if not cand:
        return None
    return max(cand, key=_stmt_key)

def _latest_txn_key(t: dict) -> str:
    return (t.get("transactionDateTime") or t.get("postedDateTime") or t.get("date") or "")

def _fmt_money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"

def _latest_statement_period_from(stmts: List[dict]):
    periods = [s.get("period") for s in stmts if isinstance(s, dict) and s.get("period")]
    return max(periods) if periods else None

def _latest_statement_period_with_interest(stmts: List[dict]):
    periods = sorted({s.get("period") for s in stmts if s.get("period")}, reverse=True)
    for p in periods:
        for s in stmts:
            if s.get("period") == p:
                val = float(s.get("interestCharged") or s.get("interest_charged") or 0)
                if val > 0: return p
    return None

def execute_calls(calls: List[dict], config_paths: dict) -> Dict[str, Any]:
    txns = load_transactions("data/folder/transactions.json")
    pays = load_payments("data/folder/payments.json")
    stmts = load_statements("data/folder/statements.json")
    acct  = load_account_summary("data/folder/account_summary.json")

    results = {}
    latest_stmt_period = _latest_statement_period_from(stmts)
    intent = (config_paths or {}).get("intent")

    for call in calls:
        dom = call.get("domain_id"); cap = call.get("capability"); args = dict(call.get("args", {}))
        key = f"{dom}.{cap}"

        if dom == "transactions":
            if cap == "list_over_threshold":
                thr = float(args.get("threshold", 0))
                res = txn_calc.list_over_threshold(txns, args.get("account_id"), thr, args.get("period"))
            elif cap == "last_transaction":
                # optional account filter
                aid = args.get("account_id")
                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                if not cand:
                    res = {"error": "No transactions", "trace": {"count": 0}}
                else:
                    last = max(cand, key=_latest_txn_key)
                    res = {"item": last, "trace": {"count": len(cand)}}

            elif cap == "top_merchants":
                # Sum purchases by merchantName, descending
                aid = args.get("account_id")
                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                # purchases only (heuristic: DEBIT or positive amount)
                rows = [t for t in cand if (t.get("transactionType") == "DEBIT" or (t.get("amount") or 0) > 0)]
                totals = {}
                for t in rows:
                    m = (t.get("merchantName") or "UNKNOWN").strip()
                    totals[m] = totals.get(m, 0.0) + float(t.get("amount") or 0.0)
                top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
                res = {
                    "top_merchants": [{"merchant": m, "total": v} for m, v in top],
                    "trace": {"count": len(rows)}
                }
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
                    txns, args.get("account_id"),
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
            elif cap == "compare_periods":
                res = txn_calc.compare_periods(txns, args.get("account_id"), args.get("period1"), args.get("period2"))
            else:
                res = {"error": f"Unknown capability {cap}"}

        elif dom == "payments":
            if cap == "last_payment":
                res = pay_calc.last_payment(pays, args.get("account_id"))
            elif cap == "total_credited_year":
                yr = args.get("year")
                res = pay_calc.total_credited_year(pays, args.get("account_id"), int(yr) if yr else None)
            elif cap == "payments_in_period":
                res = pay_calc.payments_in_period(pays, args.get("account_id"), args.get("period"))
            else:
                res = {"error": f"Unknown capability {cap}"}

        elif dom == "statements":

            # fill missing/NULL period
            nonzero = bool(args.get("nonzero"))
            if not args.get("period"):
                pick = _pick_latest_stmt(stmts, nonzero=nonzero)
                if pick:
                    args["period"] = pick.get("period")
                    args["_picked_close"] = pick.get("closingDateTime")
                    args["_picked_interest"] = pick.get("interestCharged")
                else:
                    args["period"] = None

            if not args.get("period"):
                if args.get("nonzero") or intent == "last_interest":
                    args["period"] = _latest_statement_period_with_interest(stmts) or latest_stmt_period
                else:
                    args["period"] = latest_stmt_period

            if cap == "total_interest":
                prd = args.get("period")
                if prd is None:
                    res = {"error": "No statements available", "trace": {"period": None}}
                else:
                    if args.get("_picked_interest") is not None:
                        amt = args["_picked_interest"]
                        res = {"interest_total": amt,
                               "trace": {"period": prd, "close_date": args.get("_picked_close"), "nonzero": nonzero}}
                    else:
                        # find exact period
                        row = next((s for s in stmts if s.get("period") == prd), None)
                        if not row:
                            res = {"error": "No statement for period", "trace": {"period": prd}}
                        else:
                            amt = row.get("interestCharged") or 0.0
                            res = {"interest_total": amt,
                                   "trace": {"period": prd, "close_date": row.get("closingDateTime"),
                                             "nonzero": nonzero}}
            elif cap == "interest_breakdown":
                res = stmt_calc.interest_breakdown(stmts, args.get("account_id"), args["period"])
            elif cap == "trailing_interest":
                res = stmt_calc.trailing_interest(stmts, args.get("account_id"), args["period"])
            else:
                res = {"error": f"Unknown capability {cap}"}

            if isinstance(res, dict):
                tr = res.setdefault("trace", {})
                if isinstance(tr, dict) and not tr.get("period") and args.get("period"):
                    tr["period"] = args["period"]

        elif dom == "account_summary":
            if cap == "current_balance":
                res = acct_calc.current_balance(acct)
            elif cap == "available_credit":
                res = acct_calc.available_credit(acct)
            else:
                res = {"error": f"Unknown capability {cap}"}

        elif dom == "policy":
            res = get_policy_snippet(cap)

        else:
            res = {"error": f"Unknown domain {dom}"}

        results[key] = res

    return results
