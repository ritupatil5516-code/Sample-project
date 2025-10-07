# core/orchestrator/compose.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
from core.utils.formatters import fmt_money
import calendar


# ----------------------------- format helpers --------------------------------
def _iso(dt: Optional[str]) -> str:
    """Return a human-readable date from ISO-like strings; else echo back."""
    if not dt:
        return "unknown date"
    try:
        # handle "YYYY-MM-DDTHH:MM:SSZ" or "YYYY-MM-DD"
        if "T" in dt:
            return datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        if len(dt) == 10:
            return datetime.fromisoformat(dt).strftime("%Y-%m-%d")
        # period forms like "YYYY-MM"
        if len(dt) == 7:
            y, m = dt.split("-")
            return f"{calendar.month_name[int(m)]} {y}"
        if len(dt) == 4:
            return dt
    except Exception:
        pass
    return dt


def _fmt_period(period: Optional[str]) -> str:
    if not period:
        return ""
    if len(period) == 7:
        y, m = period.split("-")
        return f"{calendar.month_name[int(m)]} {y}"
    if len(period) == 4:
        return period
    return period


def _pick_amount(obj: dict) -> float:
    for k in ("amount", "total", "value"):
        if k in obj:
            try:
                return float(obj[k])
            except Exception:
                pass
    return 0.0


def _pick_dt(obj: dict) -> Optional[str]:
    for k in ("transactionDateTime", "paymentDateTime", "closingDateTime", "date", "transaction_date_time"):
        if k in obj:
            return obj[k]
    return None


def _pick_merchant(obj: dict) -> str:
    for k in ("merchantName", "merchant_name", "merchant"):
        if k in obj and obj[k]:
            return str(obj[k])
    return "Unknown merchant"


# --------------------------- main composition --------------------------------

def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert deterministic results into a friendly answer.
    Never invent numbers—only use values returned in `results`.
    Return dict: {"answer": str, "citations": List[str]}
    """

    # ===== TRANSACTIONS: average_per_month ====================================
    # --- average_per_month (transactions) — clean, readable output ---------------
    if "transactions.average_per_month" in results:
        r = results.get("transactions.average_per_month") or {}
        months = r.get("months") or []
        avg = r.get("average", 0.0)
        n = int(r.get("count_months", 0) or 0)
        total = r.get("total", 0.0)
        period = (r.get("trace") or {}).get("period")

        def _fmt_money(x):
            try:
                return f"${float(x):,.2f}"
            except Exception:
                return "$0.00"

        import calendar

        # Build a clean list like: "• April 2025 — $1,839.99"
        month_lines = []
        for m in months:
            p = str(m.get("period", "") or "")
            amt = _fmt_money(m.get("total", 0))
            label = p
            if len(p) == 7:  # YYYY-MM
                y, mo = p.split("-")
                label = f"{calendar.month_name[int(mo)]} {y}"
            elif len(p) == 4:  # YYYY
                label = p
            month_lines.append(f"- {label} — {amt}")

        # Header line with optional period label
        ptxt = ""
        if period:
            if len(period) == 7:
                y, mo = period.split("-")
                ptxt = f" in {calendar.month_name[int(mo)]} {y}"
            elif len(period) == 4:
                ptxt = f" in {period}"

        # Final message (use explicit newlines for readability)
        lines = [
            f"Your average monthly spend{ptxt} is {_fmt_money(avg)} across {n} month(s) (total {_fmt_money(total)})."
        ]
        if month_lines:
            lines.append("")
            lines.append("By month:")
            lines.extend(month_lines)

        return {
            "answer": "\n".join(lines),
            "citations": []
        }

    # ===== TRANSACTIONS: compare_periods ======================================
    # --- compare_periods (transactions) — clean, readable output -----------------
    if "transactions.compare_periods" in results:
        r = results.get("transactions.compare_periods") or {}
        p1, p2 = r.get("period1"), r.get("period2")
        t1, t2 = r.get("total1", 0.0), r.get("total2", 0.0)
        d = r.get("delta", 0.0)

        def _fmt_money(x):
            try:
                return f"${float(x):,.2f}"
            except Exception:
                return "$0.00"

        def _pp(p):  # pretty period
            if not p:
                return "—"
            if len(p) == 7:
                y, m = p.split("-")
                import calendar
                return f"{calendar.month_name[int(m)]} {y}"
            return p

        dir_word = "more" if d > 0 else ("less" if d < 0 else "the same as")
        delta_txt = f" ({_fmt_money(abs(d))} {dir_word})" if d != 0 else ""

        lines = [
            f"Spending comparison:",
            f"- {_pp(p1)} — {_fmt_money(t1)}",
            f"- {_pp(p2)} — {_fmt_money(t2)}",
            f"",
            f"Result: {_fmt_money(t1)} in {_pp(p1)} is {dir_word} {_fmt_money(abs(d)) if d != 0 else ''} than {_pp(p2)}."
            .replace("  ", " ").strip(),
        ]
        return {"answer": "\n".join(lines), "citations": []}

    # ===== TRANSACTIONS: max_amount (biggest purchase) ========================
    if "transactions.max_amount" in results:
        r = results.get("transactions.max_amount") or {}
        items = r.get("items") or []
        if not items:
            return {"answer": "I didn’t find a qualifying purchase for that period.", "citations": []}
        top = items[0]
        amt = fmt_money(_pick_amount(top))
        merch = _pick_merchant(top)
        when = _iso(_pick_dt(top))
        period = (r.get("trace") or {}).get("period")
        cat = (r.get("trace") or {}).get("category")
        extra_cat = f" in **{cat}**" if cat else ""
        extra_p = f" ({_fmt_period(period)})" if period else ""
        return {"answer": f"Your biggest purchase{extra_cat}{extra_p} was {amt} at **{merch}** on {when}.", "citations": []}

    # ===== TRANSACTIONS: aggregate_by_category =================================
    if "transactions.aggregate_by_category" in results:
        r = results.get("transactions.aggregate_by_category") or {}
        buckets = r.get("buckets") or []
        if not buckets:
            return {"answer": "No spend found for that period.", "citations": []}
        top_line = ", ".join([f"{b['category']}: {fmt_money(b['total'])}" for b in buckets[:5]])
        period = (r.get("trace") or {}).get("period")
        ptxt = f" in {_fmt_period(period)}" if period else ""
        return {"answer": f"Spending by category{ptxt}: {top_line}.", "citations": []}

    # ===== TRANSACTIONS: spend_in_period (sum) =================================
    # --- spend_in_period (transactions) — with Top merchants leaderboard ----------
    if "transactions.spend_in_period" in results:
        r = results.get("transactions.spend_in_period") or {}
        total = r.get("value", 0.0) or r.get("total", 0.0)
        items = r.get("items") or []
        period = (r.get("trace") or {}).get("period")

        import collections
        merchant_totals = collections.defaultdict(float)

        # Sum by merchant (only DEBIT purchases unless your executor pre-filters)
        for t in items:
            merch = (t.get("merchantName") or t.get("merchant_name") or "Unknown merchant").strip()
            amt = float(t.get("amount") or 0.0)
            # If you want to exclude credits here, uncomment:
            # ind = (t.get("debitCreditIndicator") or "").upper()
            # if ind not in ("DEBIT", "D", ""):
            #     continue
            merchant_totals[merch] += amt

        header = f"You spent {fmt_money(total)} in {_fmt_period(period) if period else 'the selected period'}."

        if not items or not merchant_totals:
            return {"answer": header, "citations": []}

        # Build top-N list (3–5 is usually enough)
        top = sorted(merchant_totals.items(), key=lambda kv: kv[1], reverse=True)[:5]
        leaderboard = [f"- {m} — {fmt_money(v)}" for m, v in top]

        lines = [header, "", "Top merchants:", *leaderboard]
        return {"answer": "\n".join(lines), "citations": []}


    # ===== TRANSACTIONS: list_over_threshold ===================================
    if "transactions.list_over_threshold" in results:
        r = results.get("transactions.list_over_threshold") or {}
        items = r.get("items") or []
        if not items:
            return {"answer": "No transactions over that threshold were found.", "citations": []}
        lines = []
        for t in items:
            amt = fmt_money(_pick_amount(t))
            merch = _pick_merchant(t)
            when = _iso(_pick_dt(t))
            lines.append(f"- {when}: {amt} at **{merch}**")
        period = (r.get("trace") or {}).get("period")
        header = f"Transactions over threshold in {_fmt_period(period)}:" if period else "Transactions over threshold:"
        return {"answer": header + "\n" + "\n".join(lines), "citations": []}

    # ===== PAYMENTS: last_payment ==============================================
    if "payments.last_payment" in results:
        r = results.get("payments.last_payment") or {}
        lp = r.get("last_payment")
        if not lp:
            return {"answer": "I couldn’t find a recent payment.", "citations": []}
        amt = fmt_money(_pick_amount(lp))
        when = _iso(lp.get("paymentDateTime"))
        return {"answer": f"Your last payment was {amt} on {when}.", "citations": []}

    # ===== PAYMENTS: total_credited_year / payments_in_period ==================
    if "payments.total_credited_year" in results:
        r = results.get("payments.total_credited_year") or {}
        year = (r.get("trace") or {}).get("year")
        total = r.get("total", 0.0)
        return {"answer": f"Your total payments in {year} are {fmt_money(total)}.", "citations": []}

    if "payments.payments_in_period" in results:
        r = results.get("payments.payments_in_period") or {}
        items = r.get("items") or []
        if not items:
            return {"answer": "No payments found for that period.", "citations": []}
        lines = []
        for p in items:
            amt = fmt_money(_pick_amount(p))
            when = _iso(p.get("paymentDateTime") or p.get("paymentPostedDateTime"))
            lines.append(f"- {when}: {amt}")
        period = (r.get("trace") or {}).get("period")
        header = f"Payments in {_fmt_period(period)}:" if period else "Payments:"
        return {"answer": header + "\n" + "\n".join(lines), "citations": []}

    # ===== STATEMENTS: total_interest / interest_breakdown / trailing_interest ==
    if "statements.total_interest" in results and "statements.trailing_interest" in results:
        ti = results.get("statements.total_interest") or {}
        tr = results.get("statements.trailing_interest") or {}
        itotal = ti.get("interest_total", 0.0)
        per = (ti.get("trace") or {}).get("period")
        ttrail = tr.get("trailing_interest", 0.0)
        msg = f"Interest charged in {_fmt_period(per) if per else 'the latest period'} was {fmt_money(itotal)}."
        if ttrail:
            msg += f" Trailing interest: {fmt_money(ttrail)}."
        return {"answer": msg, "citations": []}

    if "statements.total_interest" in results:
        ti = results.get("statements.total_interest") or {}
        itotal = ti.get("interest_total", 0.0)
        per = (ti.get("trace") or {}).get("period")
        return {"answer": f"Interest charged in {_fmt_period(per) if per else 'the latest period'} was {fmt_money(itotal)}.", "citations": []}

    if "statements.interest_breakdown" in results:
        r = results.get("statements.interest_breakdown") or {}
        if r.get("error"):
            per = (r.get("trace") or {}).get("period")
            return {"answer": f"No statement found for {_fmt_period(per) if per else 'that period'}.", "citations": []}
        per = (r.get("trace") or {}).get("period")
        parts = r.get("parts") or []
        if not parts:
            return {"answer": f"No detailed breakdown available for {_fmt_period(per) if per else 'that period'}.", "citations": []}
        lines = [f"- {p.get('label','component')}: {fmt_money(p.get('amount',0))}" for p in parts]
        header = f"Interest breakdown for {_fmt_period(per)}:" if per else "Interest breakdown:"
        return {"answer": header + "\n" + "\n".join(lines), "citations": []}

    if "statements.trailing_interest" in results:
        tr = results.get("statements.trailing_interest") or {}
        per = (tr.get("trace") or {}).get("period")
        val = tr.get("trailing_interest", 0.0)
        return {"answer": f"Trailing interest for {_fmt_period(per) if per else 'the latest period'} is {fmt_money(val)}.", "citations": []}

    # ===== ACCOUNT SUMMARY: balance / available credit =========================
    if "account_summary.current_balance" in results:
        r = results.get("account_summary.current_balance") or {}
        bal = r.get("current_balance", 0.0)
        asof = r.get("as_of_date")
        tail = f" as of {_iso(asof)}" if asof else ""
        return {"answer": f"Your current balance is {fmt_money(bal)}{tail}.", "citations": []}

    if "account_summary.available_credit" in results:
        r = results.get("account_summary.available_credit") or {}
        ac = r.get("available_credit", 0.0)
        lim = r.get("credit_limit")
        tail = f" (limit {fmt_money(lim)})" if lim is not None else ""
        return {"answer": f"Your available credit is {fmt_money(ac)}{tail}.", "citations": []}

    # ===== GENERIC / FALLBACK ==================================================
    # If we reach here, show plan/results for debugging.
    # Keep the raw dictionary short in UI; main app already shows full JSON in expander.
    if results:
        # Try to summarize keys present
        keys = ", ".join(sorted(results.keys()))
        return {"answer": f"Here is what I found: {keys}. Open 'Trace & plan' for details.", "citations": []}

    return {"answer": "I couldn’t find enough information to answer that.", "citations": []}