from typing import List, Optional

def total_interest(stmts: List[dict], account_id: Optional[str], period: Optional[str]):
    match = [s for s in stmts if (not account_id or s.get("accountId") == account_id) and (not period or s.get("period") == period)]
    if not match:
        return {"interest_total": 0.0, "trace": {"count": 0, "period": period}}
    total = sum(float(s.get("interestCharged") or s.get("interest_charged") or 0) for s in match)
    return {
        "interest_total": round(total, 2),
        "trace": {"count": len(match), "period": period, "close_date": match[-1].get("closingDateTime") or match[-1].get("closing_date_time")},
    }

def interest_breakdown(stmts: List[dict], account_id: Optional[str], period: Optional[str]):
    match = [s for s in stmts if (not account_id or s.get("accountId") == account_id) and (not period or s.get("period") == period)]
    if not match:
        return {"error": "No statement for period", "trace": {"period": period}}
    stmt = match[-1]
    non_trailing = float(stmt.get("totalNonTrailingInterest") or stmt.get("total_non_trailing_interest") or 0)
    trailing = float(stmt.get("totalTrailingInterest") or stmt.get("total_trailing_interest") or 0)
    total = float(stmt.get("interestCharged") or stmt.get("interest_charged") or (non_trailing + trailing) or 0)
    return {
        "interest_total": total,
        "non_trailing": non_trailing,
        "trailing": trailing,
        "trace": {"period": period, "close_date": stmt.get("closingDateTime") or stmt.get("closing_date_time")},
    }

def trailing_interest(stmts: List[dict], account_id: Optional[str], period: Optional[str]):
    match = [s for s in stmts if (not account_id or s.get("accountId") == account_id) and (not period or s.get("period") == period)]
    if not match:
        return {"trailing_interest": 0.0, "trace": {"period": period}}
    stmt = match[-1]
    trailing = float(stmt.get("totalTrailingInterest") or stmt.get("total_trailing_interest") or 0)
    return {
        "trailing_interest": trailing,
        "trace": {"period": period, "close_date": stmt.get("closingDateTime") or stmt.get("closing_date_time")},
    }