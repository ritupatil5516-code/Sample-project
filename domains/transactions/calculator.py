from collections import defaultdict
from typing import List, Optional, Dict

from core.semantics.synonyms import guess_category


def _starts_with_period(dt: Optional[str], period: Optional[str]) -> bool:
    return bool(dt and period and dt.startswith(period))

def list_over_threshold(txns: List[dict], account_id: Optional[str], threshold: float, period: Optional[str]):
    items = []
    for t in txns:
        if account_id and t.get("accountId") != account_id: continue
        if period and not _starts_with_period(t.get("transactionDateTime") or t.get("transaction_date_time"), period): continue
        amt = float(t.get("amount") or 0)
        if amt >= threshold: items.append(t)
    total = sum(float(x.get("amount") or 0) for x in items)
    return {"items": items, "total": round(total, 2), "trace": {"count": len(items), "period": period}}

def spend_in_period(txns: List[dict], account_id: Optional[str], period: Optional[str]):
    items, total = [], 0.0
    for t in txns:
        if account_id and t.get("accountId") != account_id: continue
        if period and not _starts_with_period(t.get("transactionDateTime") or t.get("transaction_date_time"), period): continue
        amt = float(t.get("amount") or 0)
        total += amt; items.append(t)
    return {"value": round(total, 2), "items": items, "trace": {"count": len(items), "period": period}}

def _dt_starts_with(dt: Optional[str], prefix: Optional[str]) -> bool:
    return bool(dt and prefix and dt.startswith(prefix))

def _in_range(dt: Optional[str], start: Optional[str], end: Optional[str]) -> bool:
    if not dt or not start or not end: return False
    return start <= dt[:7] <= end

def _match_period(dt: Optional[str], period: Optional[str], ps: Optional[str], pe: Optional[str]) -> bool:
    if period:
        return _dt_starts_with(dt, period)
    if ps and pe:
        return _in_range(dt, ps, pe)
    return True

def _match_category(merchant: Optional[str], category: Optional[str]) -> bool:
    if not category: return True
    return guess_category(merchant) == category

def max_amount(txns: List[dict], account_id: Optional[str], period: Optional[str], period_start: Optional[str], period_end: Optional[str], category: Optional[str], top: int = 1):
    cand = []
    for t in txns:
        if account_id and t.get("accountId") != account_id: continue
        dt = t.get("transactionDateTime") or t.get("transaction_date_time")
        if not _match_period(dt, period, period_start, period_end): continue
        if not _match_category(t.get("merchantName") or t.get("merchant_name"), category): continue
        cand.append(t)
    cand.sort(key=lambda r: float(r.get("amount") or 0.0), reverse=True)
    topn = cand[: max(top, 1)]
    return {"items": topn, "trace": {"count": len(cand), "period": period or f"{period_start}..{period_end}", "category": category}}

def aggregate_by_category(txns: List[dict], account_id: Optional[str], period: Optional[str], period_start: Optional[str], period_end: Optional[str]):
    buckets = {}
    items = []
    for t in txns:
        if account_id and t.get("accountId") != account_id: continue
        dt = t.get("transactionDateTime") or t.get("transaction_date_time")
        if not _match_period(dt, period, period_start, period_end): continue
        cat = guess_category(t.get("merchantName") or t.get("merchant_name"))
        amt = float(t.get("amount") or 0.0)
        buckets[cat] = buckets.get(cat, 0.0) + amt
        items.append({**t, "_category": cat})
    out = [{"category": k, "total": round(v, 2)} for k, v in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)]
    return {"buckets": out, "items": items, "trace": {"period": period or f"{period_start}..{period_end}"}}

def compare_periods(txns: List[dict], account_id: Optional[str], p1: str, p2: str):
    def _sum(period: str) -> float:
        s = 0.0
        for t in txns:
            if account_id and t.get("accountId") != account_id: continue
            dt = t.get("transactionDateTime") or t.get("transaction_date_time")
            if _dt_starts_with(dt, period):
                s += float(t.get("amount") or 0.0)
        return round(s, 2)
    v1, v2 = _sum(p1), _sum(p2)
    return {"period1": p1, "total1": v1, "period2": p2, "total2": v2, "delta": round(v1 - v2, 2)}

def _dt_month(dt: Optional[str]) -> Optional[str]:
    # expects ISO-like e.g. "2025-04-10T..." → "2025-04"
    return dt[:7] if dt else None

def _period_match(dt: Optional[str], period: Optional[str], ps: Optional[str], pe: Optional[str]) -> bool:
    m = _dt_month(dt)
    if not m:
        return False
    if period:
        return m.startswith(period)  # period can be "YYYY" or "YYYY-MM"
    if ps and pe:
        return ps <= m <= pe
    return True  # if no period provided, accept all

def _is_debit(t: dict) -> bool:
    # treat purchases as debits (spend)
    ind = (t.get("debitCreditIndicator") or t.get("debit_credit_indicator") or "").upper()
    return ind in ("DEBIT", "D", "")

def _amount(t: dict) -> float:
    try:
        return float(t.get("amount") or 0.0)
    except Exception:
        return 0.0

# --- capability: average_per_month ---
def average_per_month(
    txns: List[dict],
    account_id: Optional[str],
    period: Optional[str] = None,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    include_credits: bool = False,
) -> Dict:
    """
    Computes average monthly spend over the selected window.
    - period: "YYYY" or "YYYY-MM"
    - or use period_start, period_end as "YYYY-MM" range
    - only DEBIT (spend) by default; set include_credits=True to include credits
    """
    buckets = defaultdict(float)  # month -> total
    considered = set()

    for t in txns:
        if account_id and t.get("accountId") != account_id:
            continue
        dt = t.get("transactionDateTime") or t.get("transaction_date_time")
        if not _period_match(dt, period, period_start, period_end):
            continue
        if not include_credits and not _is_debit(t):
            continue
        m = _dt_month(dt)
        if not m:
            continue
        buckets[m] += _amount(t)
        considered.add(m)

    months = [{"period": m, "total": round(buckets[m], 2)} for m in sorted(buckets.keys())]
    total = round(sum(buckets.values()), 2)
    n = len(considered)
    avg = round(total / n, 2) if n else 0.0

    # human-readable trace for UI
    trace_period = period or (f"{period_start}..{period_end}" if period_start and period_end else None)
    return {"months": months, "average": avg, "count_months": n, "total": total,
            "trace": {"period": trace_period, "months_counted": n}}