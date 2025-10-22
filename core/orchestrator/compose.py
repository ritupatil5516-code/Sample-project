# core/orchestrator/compose_answer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v:,.2f}"

def _fmt_dt_iso(s: Optional[str]) -> str:
    if not s or not isinstance(s, str):
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s

def _shorten(s: Any, n: int = 80) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def _pick_latest_timestamp(row: Dict[str, Any]) -> Optional[str]:
    for k in (
        "closingDateTime", "postedDateTime", "transactionDateTime",
        "paymentPostedDateTime", "paymentDateTime", "openingDateTime", "date"
    ):
        if k in row and row[k]:
            return _fmt_dt_iso(str(row[k]))
    return None

def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    return []

def _domain_from_key(key: str) -> Tuple[str, str]:
    # "statements.find_latest[0]" -> ("statements", "find_latest")
    try:
        dom, rest = key.split(".", 1)
    except ValueError:
        return "general", key
    op = rest.split("[", 1)[0]
    return dom, op

# --------------------------------------------------------------------
# Renderers (human text)
# --------------------------------------------------------------------

def _render_get_field(value: Any) -> str:
    if isinstance(value, (int, float)):
        # heuristically format money if >= $1 in magnitude
        return _fmt_money(value) if abs(float(value)) >= 1 else str(value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

def _first_present_number(row: Dict[str, Any], keys: List[str]):
    """Return first numeric value (prefer > 0); if none >0, return the first numeric even if 0."""
    first_numeric = None
    for k in keys:
        v = row.get(k)
        try:
            f = float(v)
        except Exception:
            continue
        if first_numeric is None:
            first_numeric = f
        if f > 0:
            return f
    return first_numeric

def _render_find_latest(payload: Dict[str, Any], domain: str) -> str:
    row = payload.get("row") or {}
    ts = _pick_latest_timestamp(row)
    if domain == "statements":
        # Try common interest keys
        interest_keys = [
            "interestCharged", "totalInterestCharged", "interestAmount",
            "financeCharge", "totalFinanceCharge", "totalInterest", "totalTrailingInterest"
        ]
        amt = _first_present_number(row, interest_keys)
        date = row.get("closingDateTime") or ts
        s_date = _fmt_dt_iso(date) if date else ""

        if amt is None:
            # no recognizable interest fields
            return f"Latest statement{(' on ' + s_date) if s_date else ''}; no interest field found."
        if amt > 0:
            return f"You were charged interest of {_fmt_money(amt)}" + (f" on {s_date}." if s_date else ".")
        # amt == 0 → clarify
        return f"Latest statement{(' on ' + s_date) if s_date else ''} shows $0.00 interest."
    # generic fallback
    head = _render_get_field(payload.get("value"))
    return f"{head}" + (f" (as of {ts})" if ts else "")

def _render_sum_where(payload: Dict[str, Any]) -> str:
    total = payload.get("total")
    cnt = payload.get("count")
    t = _fmt_money(total) if isinstance(total, (int, float)) else str(total)
    return f"{t}" + (f" across {cnt} items" if isinstance(cnt, int) else "")

def _render_topk_by_sum(payload: Dict[str, Any]) -> str:
    rows = payload.get("top") or []
    if not rows: return "No results."
    lines = []
    for i, r in enumerate(rows, 1):
        key = r.get("key", "UNKNOWN")
        tot = r.get("total", 0)
        lines.append(f"{i}. {key}: {_fmt_money(tot)}")
    return "\n".join(lines)

def _render_list_where(payload: Dict[str, Any]) -> str:
    items = _as_list(payload.get("items"))
    if not items: return "No matching items."
    # choose 5 columns that are most common in this domain
    cols_pref = [
        "postedDateTime", "transactionDateTime", "paymentPostedDateTime",
        "merchantName", "description", "amount", "transactionStatus",
        "displayTransactionType", "category", "period"
    ]
    sample = items[0]
    cols = [c for c in cols_pref if c in sample][:4] or list(sample.keys())[:4]
    # table
    header = " | ".join(cols)
    sep = " | ".join("---" for _ in cols)
    lines = [header, sep]
    for r in items[:15]:
        line = " | ".join(
            _shorten(
                _fmt_money(r.get(c)) if c == "amount"
                else (_fmt_dt_iso(r.get(c)) if c in ("postedDateTime","transactionDateTime","paymentPostedDateTime","closingDateTime") else r.get(c))
            )
            for c in cols
        )
        lines.append(line)
    extra = len(items) - min(15, len(items))
    if extra > 0:
        lines.append(f"... and {extra} more")
    return "\n".join(lines)

# Legacy calculator fallbacks
def _render_legacy_value(payload: Dict[str, Any]) -> Optional[str]:
    if "interest_total" in payload:
        return _fmt_money(payload["interest_total"])
    if "item" in payload and isinstance(payload["item"], dict):
        r = payload["item"]
        ts = _pick_latest_timestamp(r)
        amt = r.get("amount")
        m   = r.get("merchantName") or r.get("description") or ""
        s_amt = f" for {_fmt_money(amt)}" if amt is not None else ""
        s_ts = f" on {ts}" if ts else ""
        s_m = f" at {m}" if m else ""
        return f"Latest transaction{s_m}{s_amt}{s_ts}".strip()
    if "top_merchants" in payload:
        rows = payload["top_merchants"]
        if not rows: return "No results."
        return "\n".join(f"{i+1}. {r.get('merchant','UNKNOWN')}: {_fmt_money(r.get('total',0))}"
                         for i, r in enumerate(rows))
    if "total" in payload and isinstance(payload["total"], (int, float)):
        return _fmt_money(payload["total"])
    if "items" in payload and isinstance(payload["items"], list):
        return _render_list_where({"items": payload["items"]})
    return None

# --------------------------------------------------------------------
# Struct builders (JSON answer items)
# --------------------------------------------------------------------

def _struct_for_get_field(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": domain,
        "capability": "get_field",
        "value": payload.get("value"),
        "trace": payload.get("trace") or {}
    }

def _struct_for_find_latest(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    row = payload.get("row") or {}
    ts = _pick_latest_timestamp(row)
    out = {
        "domain": domain,
        "capability": "find_latest",
        "value": payload.get("value"),
        "row": row,
        "timestamp": ts,
        "trace": payload.get("trace") or {}
    }
    if domain == "statements":
        out["interestCharged"] = row.get("interestCharged")
        out["closingDateTime"] = row.get("closingDateTime")
    return out

def _struct_for_sum_where(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": domain,
        "capability": "sum_where",
        "total": payload.get("total"),
        "count": payload.get("count"),
        "trace": payload.get("trace") or {}
    }

def _struct_for_topk_by_sum(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": domain,
        "capability": "topk_by_sum",
        "top": payload.get("top") or [],
        "trace": payload.get("trace") or {}
    }

def _struct_for_list_where(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": domain,
        "capability": "list_where",
        "items": _as_list(payload.get("items")),
        "aggregate": payload.get("aggregate"),
        "trace": payload.get("trace") or {}
    }

def _struct_for_semantic_search(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": domain,
        "capability": "semantic_search",
        "hits": payload.get("hits") or [],
        "trace": payload.get("trace") or {}
    }

# --------------------------------------------------------------------
# Public compose
# --------------------------------------------------------------------

def compose_answer(
    question: str,
    plan: Dict[str, Any],
    results: Dict[str, Any],
    return_struct: bool = False
) -> Any:
    """
    Formats heterogeneous `results` into:
      - default: a single human-readable string (backward compatible)
      - if return_struct=True: a dict with {"answer": <text>, "intent": str, "strategy": str, "items": [...]}
    """

    if not results:
        return {"answer": "I couldn't find anything for that.", "intent": (plan or {}).get("intent"), "strategy": (plan or {}).get("strategy"), "items": []} if return_struct else "I couldn't find anything for that."

    # Human text lines + structured items
    lines: List[str] = []
    items: List[Dict[str, Any]] = []

    intent = (plan or {}).get("intent")
    strategy = (plan or {}).get("strategy")

    # Iterate each result bucket
    for key, payload in results.items():
        domain, op = _domain_from_key(key)
        p = payload if isinstance(payload, dict) else {}

        if domain == "rag" or op in ("unified_answer", "account_answer", "handbook_answer", "knowledge_answer"):
            lines.append(_render_rag(payload if isinstance(payload, dict) else {}))
            continue

        # Human text + struct
        if op == "get_field":
            txt = _render_get_field(p.get("value"))
            items.append(_struct_for_get_field(domain, p))
        elif op == "find_latest":
            txt = _render_find_latest(p, domain)
            items.append(_struct_for_find_latest(domain, p))
        elif op == "sum_where":
            txt = _render_sum_where(p)
            items.append(_struct_for_sum_where(domain, p))
        elif op == "topk_by_sum":
            txt = _render_topk_by_sum(p)
            items.append(_struct_for_topk_by_sum(domain, p))
        elif op == "list_where":
            txt = _render_list_where(p)
            items.append(_struct_for_list_where(domain, p))
        elif op == "explain_interest":
            txt = _render_explain_interest(p)
            items.append(_struct_for_explain_interest(domain, p))
        elif op == "semantic_search":
            hits = p.get("hits") or []
            if not hits:
                txt = "No relevant matches found."
            else:
                view = []
                for h in hits[:5]:
                    pay = h.get("payload") or {}
                    ts = _pick_latest_timestamp(pay) or ""
                    amt = pay.get("amount")
                    m   = pay.get("merchantName") or ""
                    piece = _shorten(h.get("text") or m or "match")
                    s = piece
                    if amt is not None: s += f" · {_fmt_money(amt)}"
                    if ts: s += f" · {ts}"
                    view.append(s)
                txt = "\n".join(view)
            items.append(_struct_for_semantic_search(domain, p))
        else:
            rend = _render_legacy_value(p)
            txt = rend if rend is not None else _shorten(json.dumps(p, ensure_ascii=False))

        if txt and str(txt).strip():
            lines.append(txt)

    # Prefer a single-sentence smart summary for common cases
    answer_text = "\n\n".join(l for l in lines if l and str(l).strip())
    if not answer_text.strip():
        answer_text = "I couldn't find anything for that."

    if return_struct:
        return {
            "answer": answer_text,
            "intent": intent,
            "strategy": strategy,
            "items": items
        }
    return answer_text


def _render_rag(payload: Dict[str, Any]) -> str:
    txt = (payload or {}).get("answer") or "I couldn't find anything relevant."
    srcs = (payload or {}).get("sources") or []
    if not srcs:
        return txt
    lines = [txt, "", "Sources:"]
    for s in srcs[:5]:
        src = s.get("source") or "source"
        snip = _shorten(s.get("snippet") or "", 160)
        lines.append(f"- {src}: {snip}")
    return "\n".join(lines)

def _render_explain_interest(p: Dict[str, Any]) -> str:
    st = p.get("statement", {})
    pr = p.get("period", {})
    drivers = p.get("drivers", {})
    s_end   = _fmt_dt_iso(st.get("closingDateTime") or pr.get("end") or "")
    total   = _fmt_money(st.get("interestCharged"))
    trail   = float(st.get("trailingInterest") or 0.0)
    nontr   = float(st.get("nonTrailingInterest") or 0.0)

    parts = [f"You were charged {total} on {s_end}."]
    if trail > 0:
        parts.append(f"Of this, {_fmt_money(trail)} was trailing interest and {_fmt_money(nontr)} was new-cycle interest.")
    cb = drivers.get("carried_balance_estimate", 0.0)
    if cb:
        parts.append(f"You carried about {_fmt_money(cb)} from the prior cycle.")
    if drivers.get("purchases_in_period", 0.0):
        parts.append(f"Purchases in the cycle totalled {_fmt_money(drivers['purchases_in_period'])}.")
    if drivers.get("payments_in_period", 0.0):
        parts.append(f"Payments in the cycle totalled {_fmt_money(drivers['payments_in_period'])}.")
    if drivers.get("interest_transactions_total", 0.0):
        parts.append(f"Interest transactions posted: {_fmt_money(drivers['interest_transactions_total'])} (see sources).")
    return " ".join(parts)

def _struct_for_explain_interest(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": domain,
        "capability": "explain_interest",
        **{k: payload.get(k) for k in ("period", "statement", "drivers", "support", "trace")}
    }