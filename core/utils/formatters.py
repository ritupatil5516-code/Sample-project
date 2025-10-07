# core/utils/formatters.py
from typing import Any

def fmt_money(x: Any) -> str:
    """
    Safely format any numeric value as currency.
    Returns '$0.00' on failure.
    """
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"

def fmt_date(x: Any) -> str:
    """
    Format datetime-like strings to YYYY-MM-DD.
    """
    if not x:
        return ""
    try:
        return str(x)[:10]
    except Exception:
        return str(x)