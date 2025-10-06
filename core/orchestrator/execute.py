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
            elif cap == "spend_in_period":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))
            elif cap == "purchases_in_cycle":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))
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
            if not args.get("period"):
                if args.get("nonzero") or intent == "last_interest":
                    args["period"] = _latest_statement_period_with_interest(stmts) or latest_stmt_period
                else:
                    args["period"] = latest_stmt_period

            if not args.get("period"):
                res = {"error": "No statements available to determine period", "trace": {"period": None}}
            elif cap == "total_interest":
                res = stmt_calc.total_interest(stmts, args.get("account_id"), args["period"])
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
            res = get_policy_snippet(cap, config_paths.get("policy_store_dir"))

        else:
            res = {"error": f"Unknown domain {dom}"}

        results[key] = res

    return results
