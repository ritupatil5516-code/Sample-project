from typing import List, Optional
from datetime import datetime, timezone

def _parse_date(dt: Optional[str]):
    try:
        if dt and dt.endswith("Z"): return datetime.fromisoformat(dt.replace("Z","+00:00"))
        return datetime.fromisoformat(dt) if dt else None
    except Exception:
        return None

def last_payment(payments: List[dict], account_id: Optional[str]):
    filt = [p for p in payments if (not account_id or p.get("accountId") == account_id)]
    if not filt: return {"last_payment": None, "trace": {"count": 0}}
    last = max(filt, key=lambda p: _parse_date(p.get("paymentDateTime") or p.get("payment_date_time")) or datetime.min.replace(tzinfo=timezone.utc))
    return {"last_payment": last, "trace": {"count": len(filt)}}

def total_credited_year(payments: List[dict], account_id: Optional[str], year: Optional[int]):
    if not year: return {"total": 0.0, "trace": {"year": None}}
    total = 0.0
    for p in payments:
        if account_id and p.get("accountId") != account_id: continue
        dt = _parse_date(p.get("paymentDateTime") or p.get("payment_date_time"))
        if dt and dt.year == year: total += float(p.get("amount") or 0)
    return {"total": round(total, 2), "trace": {"year": year}}

def payments_in_period(payments: List[dict], account_id: Optional[str], period: Optional[str]):
    total = 0.0; items = []
    for p in payments:
        if account_id and p.get("accountId") != account_id: continue
        dt = p.get("paymentDateTime") or p.get("payment_date_time") or ""
        if period and not dt.startswith(period): continue
        total += float(p.get("amount") or 0); items.append(p)
    return {"total": round(total, 2), "items": items, "trace": {"period": period}}
