# this is like how to think
from __future__ import annotations
from typing import Dict

def build_hint_for_question(q: str) -> Dict[str, str]:
    """
    Lightweight, rule-of-thumb hints the planner can use to disambiguate.
    Returns a dict with a short 'content' string you can inject as a system msg.
    Safe to no-op if nothing matches.
    """
    ql = (q or "").strip().lower()
    hints = []

    # “last / latest / recent” nudges:
    if any(w in ql for w in ("last ", "latest", "recent")) and "transaction" in ql:
        hints.append("Interpret 'last transaction' as the record with the maximum transactionDateTime.")

    if any(w in ql for w in ("last ", "latest", "recent")) and "interest" in ql:
        hints.append("When user asks for last/latest interest, pick the most recent statement with non-zero interestCharged; if none, use most recent period.")

    # Average monthly spend:
    if "average" in ql and ("month" in ql or "monthly" in ql) and ("spend" in ql or "purchase" in ql):
        hints.append("Average monthly spend = mean of monthly totals grouped by YYYY-MM over available transactions.")

    # Compare months:
    if "compare" in ql and "spend" in ql:
        hints.append("To compare months, compute total spend per YYYY-MM and present a side-by-side delta.")

    # Available credit / current balance:
    if "available credit" in ql or "current balance" in ql:
        hints.append("Use account_summary.json for current_balance and available_credit; do not invent numbers.")

    # Payments:
    if "last payment" in ql or ("when" in ql and "payment" in ql):
        hints.append("Last payment = payment with maximum paymentDateTime or paymentPostedDateTime.")

    # Fallback if nothing triggered:
    if not hints:
        return {"role": "system", "content": "No special hint."}

    return {"role": "system", "content": "Hints:\n- " + "\n- ".join(hints)}