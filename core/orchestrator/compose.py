# core/orchestrator/compose.py
from __future__ import annotations
from typing import Dict, Any, Optional

def _fmt_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"

def _fmt_iso_date(iso: Optional[str]) -> str:
    """Accepts ISO like 2025-09-05T09:01:00Z or 2025-09-05; returns YYYY-MM-DD if possible."""
    if not iso:
        return "unknown date"
    s = str(iso)
    # Prefer date portion
    if "T" in s:
        return s.split("T", 1)[0]
    return s

def _fmt_period(ym: Optional[str]) -> str:
    """Accepts 'YYYY-MM' or 'YYYY-MM-DD', returns 'Mon YYYY' or the raw string if unknown."""
    if not ym:
        return "unknown period"
    s = ym[:7]  # YYYY-MM
    year = s[:4]
    try:
        m = int(s[5:7])
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        if 1 <= m <= 12:
            return f"{months[m-1]} {year}"
    except Exception:
        pass
    return ym

def _first_present(results: Dict[str, Any], keys: list[str]) -> Optional[str]:
    for k in keys:
        if k in results:
            return k
    return None

def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn deterministic result dicts into readable answers.
    Returns: {"answer": str, "citations": Optional[list[str]]}
    """
    # -------------- 1) transactions.last_transaction -------------------------
    k = "transactions.semantic_search"
    if k in results:
        data = results[k] or {}
        hits = data.get("hits") or []
        if not hits:
            return {"answer": "No information found."}
        lines = []
        for h in hits[:10]:
            p = h.get("payload") or {}
            dt = (p.get("transactionDateTime") or p.get("postedDateTime") or p.get("date") or "unknown date")
            amt = _fmt_money(p.get("amount"))
            merch = p.get("merchantName") or "UNKNOWN"
            lines.append(f"- {dt.split('T')[0]}: {amt} at {merch}")
        return {"answer": "Here’s what matched:\n\n" + "\n".join(lines)}

    k = "transactions.last_transaction"
    if k in results:
        data = results[k]
        it = (data or {}).get("item") or {}
        if not it:
            return {"answer": "I couldn’t find any transactions."}

        ts = it.get("transactionDateTime") or it.get("postedDateTime") or it.get("date")
        date = _fmt_iso_date(ts)
        amt = _fmt_money(it.get("amount"))
        merch = it.get("merchantName") or it.get("merchant") or "UNKNOWN"
        line = f"Your most recent transaction was on {date}: {amt} at {merch}."
        return {"answer": line}

    # -------------- 2) transactions.top_merchants ----------------------------
    k = "transactions.top_merchants"
    if k in results:
        data = results[k] or {}
        tops = (data.get("top_merchants") or [])[:5]
        if not tops:
            return {"answer": "I didn’t find any merchant spend in that period."}
        lines = [f"- {row['merchant']}: {_fmt_money(row['total'])}" for row in tops]
        return {"answer": "You spent the most at:\n\n" + "\n".join(lines)}

    k = "transactions.find_by_merchant"
    if k in results:
        data = results[k] or {}
        items = data.get("items") or []
        if not items:
            mq = (data.get("trace") or {}).get("merchant_query") or "that merchant"
            return {"answer": f"I didn’t find any purchases from {mq} in that period."}
        lines = []
        for t in items[:20]:
            date = _fmt_iso_date(t.get("transactionDateTime") or t.get("postedDateTime") or t.get("date"))
            amt = _fmt_money(t.get("amount"))
            merch = t.get("merchantName") or "UNKNOWN"
            lines.append(f"- {date}: {amt} at {merch}")
        return {"answer": "Here’s what I found:\n\n" + "\n".join(lines)}

    # -------------- 3) statements.total_interest -----------------------------
    k = "statements.total_interest"
    if k in results:
        data = results[k] or {}
        amt = _fmt_money((data.get("interest_total") or 0))
        tr = data.get("trace") or {}
        period = _fmt_period(tr.get("period"))
        line = f"Your interest for {period} was {amt}."
        # If this was a 'nonzero' search, make that explicit:
        if tr.get("nonzero"):
            line = f"Your last interest charge (most recent non-zero) was {amt} in {period}."
        return {"answer": line}

    # -------------- 4) account_summary.current_balance -----------------------
    k = "account_summary.current_balance"
    if k in results:
        data = results[k] or {}
        bal = _fmt_money(data.get("current_balance"))
        dt = data.get("as_of_date")
        suffix = f" as of {_fmt_iso_date(dt)}" if dt else ""
        return {"answer": f"Your current balance is {bal}{suffix}."}

    # -------------- 5) account_summary.available_credit ----------------------
    k = "account_summary.available_credit"
    if k in results:
        data = results[k] or {}
        lim = _fmt_money(data.get("credit_limit"))
        avail = _fmt_money(data.get("available_credit") or data.get("available"))
        tail = []
        if data.get("account_last4"):
            tail.append(f"account ending {data['account_last4']}")
        line = f"Your available credit is {avail} (limit {lim})"
        if tail:
            line += f" for {'; '.join(tail)}"
        return {"answer": line + "."}

    # -------------- 6) payments.last_payment --------------------------------
    k = "payments.last_payment"
    if k in results:
        data = results[k] or {}
        lp = data.get("last_payment") or {}
        if not lp:
            return {"answer": "I couldn’t find any payments."}
        dt = _fmt_iso_date(lp.get("paymentDateTime") or lp.get("paymentPostedDateTime"))
        amt = _fmt_money(lp.get("amount"))
        return {"answer": f"Your last payment was {amt} on {dt}."}

    # -------------- 7) transactions.list_over_threshold ----------------------
    k = "transactions.list_over_threshold"
    if k in results:
        data = results[k] or {}
        items = data.get("items") or []
        if not items:
            return {"answer": "No transactions matched that threshold."}
        lines = []
        for t in items[:20]:  # keep it tight
            date = _fmt_iso_date(t.get("transactionDateTime") or t.get("postedDateTime") or t.get("date"))
            amt = _fmt_money(t.get("amount"))
            merch = t.get("merchantName") or t.get("merchant") or "UNKNOWN"
            lines.append(f"{date}: {amt} at {merch}")
        return {"answer": "Transactions over threshold:\n\n" + "\n".join(lines)}

    # -------------- 8) transactions.spend_in_period --------------------------
    k = "transactions.spend_in_period"
    if k in results:
        data = results[k] or {}
        total = _fmt_money(data.get("value") or data.get("total"))
        pr = _fmt_period((data.get("trace") or {}).get("period"))
        return {"answer": f"You spent {total} in {pr}."}

    # -------------- 9) payments.total_credited_year --------------------------
    k = "payments.total_credited_year"
    if k in results:
        data = results[k] or {}
        total = _fmt_money(data.get("total") or data.get("value"))
        yr = (data.get("trace") or {}).get("year")
        if yr:
            return {"answer": f"Total credited in {yr} was {total}."}
        return {"answer": f"Total credited was {total}."}

    # -------------- 10) payments.payments_in_period --------------------------
    k = "payments.payments_in_period"
    if k in results:
        data = results[k] or {}
        items = data.get("items") or []
        if not items:
            return {"answer": "No payments in that period."}
        pr = (data.get("trace") or {}).get("period")
        line = [f"Payments in {_fmt_period(pr)}:"]
        for p in items[:20]:
            line.append(f"- {_fmt_iso_date(p.get('paymentDateTime') or p.get('paymentPostedDateTime'))}: {_fmt_money(p.get('amount'))}")
        return {"answer": "\n".join(line)}

    # -------------- Fallback --------------------------------------------------
    # Find any domain key and present a neutral message to remind to open Trace.
    any_key = _first_present(results, list(results.keys()))
    if any_key:
        return {"answer": f"Here is what I found: {any_key}. Open 'Trace & plan' for details."}

    # ---------------------------------------------------------------------
    # 🧩 Universal fallback for missing or invalid responses
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Universal fallback for missing/invalid responses
    # ---------------------------------------------------------------------
    try:
        if not results:
            return {"answer": "No information found."}
        # If all result entries empty or errored
        valid = [k for k, v in results.items() if v and not v.get("error")]
        if not valid:
            return {"answer": "No information found."}
        if not plan or not plan.get("calls"):
            return {"answer": "No information found."}
    except Exception:
        return {"answer": "No information found."}

    # If nothing matched any case, don’t show generic “Here’s what I found…”
    return {"answer": "No information found."}