from typing import List, Optional

def _starts_with_period(dt: Optional[str], period: Optional[str]) -> bool:
    return bool(dt and period and dt.startswith(period))

def list_over_threshold(txns: List[dict], account_id: Optional[str], threshold: float, period: Optional[str]):
    items = []
    for t in txns:
        if account_id and t.get("accountId") != account_id:
            continue
        if period and not _starts_with_period(t.get("transactionDateTime") or t.get("transaction_date_time"), period):
            continue
        amt = float(t.get("amount") or 0)
        if amt >= threshold:
            items.append(t)
    total = sum(float(x.get("amount") or 0) for x in items)
    return {"items": items, "total": round(total, 2), "trace": {"count": len(items), "period": period}}

def spend_in_period(txns: List[dict], account_id: Optional[str], period: Optional[str]):
    items = []
    total = 0.0
    for t in txns:
        if account_id and t.get("accountId") != account_id:
            continue
        if period and not _starts_with_period(t.get("transactionDateTime") or t.get("transaction_date_time"), period):
            continue
        amt = float(t.get("amount") or 0)
        total += amt
        items.append(t)
    return {"value": round(total, 2), "items": items, "trace": {"count": len(items), "period": period}}