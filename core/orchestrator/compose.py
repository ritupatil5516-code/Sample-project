from typing import Dict, Any, List, Optional
from datetime import datetime

def _iso(dt: Optional[str]) -> Optional[str]:
    if not dt: return None
    try:
        if dt.endswith("Z"):
            return datetime.fromisoformat(dt.replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M:%S %Z")
        return datetime.fromisoformat(dt).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt

def _fmt_money(x: Optional[float]) -> str:
    try: return f"${float(x):,.2f}"
    except Exception: return f"${x}"

def _pick_dt(rec: dict) -> Optional[str]:
    return rec.get("transactionDateTime") or rec.get("transaction_date_time")         or rec.get("paymentDateTime") or rec.get("payment_date_time")         or rec.get("closingDateTime") or rec.get("closing_date_time")

def _pick_merchant(rec: dict) -> str:
    return rec.get("merchantName") or rec.get("merchant_name") or rec.get("displayTransactionType") or "Unknown merchant"

def _pick_amount(rec: dict) -> float:
    try: return float(rec.get("amount") or 0.0)
    except Exception: return 0.0

def _last_transaction_answer(items: List[dict]) -> Dict[str, Any]:
    if not items: return {"answer": "I couldn’t find any transactions.", "citations": []}
    def _key(rec):
        dt = _pick_dt(rec)
        try:
            if dt and dt.endswith("Z"): from datetime import datetime; return datetime.fromisoformat(dt.replace("Z","+00:00"))
            from datetime import datetime; return datetime.fromisoformat(dt) if dt else datetime.min
        except Exception:
            from datetime import datetime; return datetime.min
    last = sorted(items, key=_key, reverse=True)[0]
    when = _iso(_pick_dt(last)); merch = _pick_merchant(last); amt = _fmt_money(_pick_amount(last))
    return {"answer": f"Your most recent transaction was {amt} at **{merch}** on {when}.", "citations": []}

def _top_merchant_answer(items: List[dict]) -> Dict[str, Any]:
    if not items: return {"answer": "I didn’t find any spending in that period.", "citations": []}
    by_merchant = {}
    for r in items:
        m = _pick_merchant(r); by_merchant[m] = by_merchant.get(m, 0.0) + _pick_amount(r)
    top_m, top_val = max(by_merchant.items(), key=lambda kv: kv[1])
    total = sum(by_merchant.values())
    top3 = sorted(by_merchant.items(), key=lambda kv: kv[1], reverse=True)[:3]
    breakdown = "; ".join([f"{m}: {_fmt_money(v)}" for m, v in top3])
    return {"answer": (f"You spent the most at **{top_m}**: {_fmt_money(top_val)}.\n\n"
                       f"Top merchants: {breakdown}.\n"
                       f"Total considered: {_fmt_money(total)}."), "citations": []}

def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    if not results: return {"answer": "I couldn’t find relevant data.", "citations": []}

    if "account_summary.current_balance" in results:
        r = results["account_summary.current_balance"] or {}
        return {"answer": f"Your current balance is {_fmt_money(r.get('current_balance', 0))} (as of {r.get('as_of_date')}).", "citations": []}

    if "account_summary.available_credit" in results:
        r = results["account_summary.available_credit"] or {}
        return {"answer": f"Available credit is {_fmt_money(r.get('available_credit', 0))}; credit limit is {_fmt_money(r.get('credit_limit', 0))}.", "citations": []}

    if "statements.total_interest" in results:
        r = results["statements.total_interest"] or {}
        per = (r.get("trace") or {}).get("period")
        return {"answer": f"Interest charged in **{per}** was {_fmt_money(r.get('interest_total', 0))}.", "citations": []}

    if "transactions.list_over_threshold" in results:
        r = results["transactions.list_over_threshold"] or {}
        items = r.get("items") or []
        return _last_transaction_answer(items)

    if "transactions.spend_in_period" in results:
        r = results["transactions.spend_in_period"] or {}
        items = r.get("items") or []
        return _top_merchant_answer(items)

    if "payments.last_payment" in results:
        r = results["payments.last_payment"] or {}
        p = r.get("last_payment") or {}
        amt = _fmt_money(p.get("amount"))
        dt = _iso(p.get("paymentDateTime"))
        return {"answer": f"Your last payment was {amt} on {dt}.", "citations": []}

    return {"answer": f"Here is what I found:\n{results}", "citations": []}
