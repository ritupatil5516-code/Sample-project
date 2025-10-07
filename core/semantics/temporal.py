# core/semantics/temporal.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import re

def this_year(today: datetime) -> str:
    return f"{today.year}"

def this_month(today: datetime) -> str:
    return f"{today.year}-{today.month:02d}"

def last_month(today: datetime) -> str:
    m = today.month - 1 or 12
    y = today.year if today.month > 1 else today.year - 1
    return f"{y}-{m:02d}"

def last_summer(today: datetime) -> tuple[str, str]:
    # Jun–Aug of the most recent summer (assumes northern hemisphere)
    y = today.year if today.month >= 9 else today.year - 1
    return (f"{y}-06", f"{y}-08")

def resolve_period(text: str, today: datetime | None = None) -> dict:
    """
    Returns one of:
      {"period": "YYYY"} or {"period": "YYYY-MM"} or {"period_start": "YYYY-MM", "period_end": "YYYY-MM"} or {}
    """
    today = today or datetime.now(timezone.utc)
    t = text.lower()

    # explicit YYYY-MM
    m = re.search(r"(20\d{2})-(0[1-9]|1[0-2])", t)
    if m:
        return {"period": f"{m.group(1)}-{m.group(2)}"}

    # explicit YYYY
    m = re.search(r"(20\d{2})", t)
    if m:
        return {"period": m.group(1)}

    # semantic
    if "this month" in t:
        return {"period": this_month(today)}
    if "last month" in t:
        return {"period": last_month(today)}
    if "this year" in t or "year to date" in t or "ytd" in t:
        return {"period": this_year(today)}
    if "last summer" in t:
        start, end = last_summer(today)
        return {"period_start": start, "period_end": end}

    # “latest” / “recent” handled in executor (fills from statements)
    return {}