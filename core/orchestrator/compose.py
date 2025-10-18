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

def _render_find_latest(payload: Dict[str, Any], domain: str) -> str:
    """
    Special phrasing for statements interest questions:
    - If domain == 'statements' and row has interestCharged, include amount + closing date.
    Generic fallback otherwise.
    """
    row = payload.get("row") or {}
    val = payload.get("value")
    ts = _pick_latest_timestamp(row)
    if domain == "statements":
        amt = row.get("interestCharged")
        if amt is not None:
            date = row.get("closingDateTime") or ts
            s_date = _fmt_dt_iso(date) if date else ""
            return f"You were charged interest of {_fmt_money(amt)}" + (f" on {s_date}." if s_date else ".")
    # generic phrasing
    head = _render_get_field(val)
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

        # Human text
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
            # Legacy fallback
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