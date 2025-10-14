# core/orchestrator/compose_answer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

"""
Compose a final user-visible answer from heterogeneous results produced by execute_calls.
Supported payload shapes:
  - DSL ops:
      get_field -> {"value": <any>} OR {"values": [<any>], "count": <int>}
      find_latest -> {"value": <any>, "row": {...}}
      sum_where -> {"total": <number>, "count": <int>}
      topk_by_sum -> {"top":[{"key":str,"total":number}, ...]}
      list_where -> {"items":[{...}, ...]}
      semantic_search -> {"hits":[{"text":str,"payload":{...},"score":float}, ...]}
  - RAG:
      {"answer": str, "sources":[{"source": str, "snippet": str}, ...]}
  - Legacy calculators:
      {"interest_total": number}
      {"item": {...}}  # e.g., last_transaction
      {"top_merchants": [{"merchant":str,"total":number}, ...]}
      {"total": number}
      {"items": [ ... ]}
"""

# ------------------ small helpers ------------------

def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v:,.2f}"

def _fmt_dt_iso(s: Optional[str]) -> str:
    if not s or not isinstance(s, str): return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s

def _shorten(s: Any, n: int = 90) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def _pick_latest_timestamp(row: Dict[str, Any]) -> Optional[str]:
    for k in (
        "postedDateTime", "transactionDateTime", "paymentPostedDateTime",
        "paymentDateTime", "closingDateTime", "openingDateTime", "date"
    ):
        if k in row and row[k]:
            return _fmt_dt_iso(str(row[k]))
    return None

def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    return []

def _is_scalar(x: Any) -> bool:
    return not isinstance(x, (dict, list))

def _fmt_scalar(x: Any) -> str:
    try:
        f = float(x)
        return _fmt_money(f) if abs(f) >= 1.0 else str(x)
    except Exception:
        return str(x)

# ------------------ renderers per op ------------------

def _render_get_field_scalar(value: Any) -> str:
    if isinstance(value, (int, float)):
        return _fmt_money(value) if abs(float(value)) >= 1 else str(value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

def _render_get_field_series(values: List[Any], total_count: Optional[int] = None) -> str:
    if not values:
        return "No results."
    if all(_is_scalar(v) for v in values):
        formatted = [_fmt_scalar(v) for v in values]
        if len(formatted) <= 6:
            return ", ".join(formatted)
        lines = [f"- {v}" for v in formatted[:15]]
        extra = (total_count or len(formatted)) - min(15, len(formatted))
        if extra > 0:
            lines.append(f"... and {extra} more")
        return "\n".join(lines)
    lines = []
    max_lines = 10
    for v in values[:max_lines]:
        if isinstance(v, (dict, list)):
            lines.append(_shorten(json.dumps(v, ensure_ascii=False), 120))
        else:
            lines.append(_shorten(_fmt_scalar(v), 120))
    extra = (total_count or len(values)) - min(max_lines, len(values))
    if extra > 0:
        lines.append(f"... and {extra} more")
    return "\n".join(lines)

def _render_get_field_payload(payload: Dict[str, Any]) -> str:
    if "value" in payload:
        return _render_get_field_scalar(payload.get("value"))
    if "values" in payload:
        vals = payload.get("values") or []
        total = payload.get("count")
        if not isinstance(vals, list):
            return _render_get_field_scalar(vals)
        if len(vals) == 1 and _is_scalar(vals[0]):
            return _render_get_field_scalar(vals[0])
        return _render_get_field_series(vals, total)
    # last resort
    return _shorten(json.dumps(payload, ensure_ascii=False))

def _render_find_latest(payload: Dict[str, Any]) -> str:
    val = payload.get("value")
    row = payload.get("row") or {}
    ts = _pick_latest_timestamp(row)
    head = _render_get_field_scalar(val)
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
    # choose 4 columns that usually make sense across domains
    cols_pref = [
        "postedDateTime", "transactionDateTime", "paymentPostedDateTime", "paymentDateTime",
        "merchantName", "description", "amount", "transactionStatus",
        "displayTransactionType", "category", "period"
    ]
    sample = items[0]
    cols = [c for c in cols_pref if c in sample][:4] or list(sample.keys())[:4]
    header = " | ".join(cols)
    sep = " | ".join("---" for _ in cols)
    lines = [header, sep]
    for r in items[:15]:
        row_vals = []
        for c in cols:
            v = r.get(c)
            if c == "amount":
                row_vals.append(_fmt_money(v))
            elif c in ("postedDateTime", "transactionDateTime", "paymentPostedDateTime", "paymentDateTime"):
                row_vals.append(_fmt_dt_iso(v))
            else:
                row_vals.append(_shorten(v))
        lines.append(" | ".join(row_vals))
    extra = len(items) - min(15, len(items))
    if extra > 0:
        lines.append(f"... and {extra} more")
    return "\n".join(lines)

def _render_semantic_search(payload: Dict[str, Any]) -> str:
    hits = (payload or {}).get("hits") if isinstance(payload, dict) else []
    if not hits:
        return "No relevant matches found."
    view = []
    for h in hits[:5]:
        p = h.get("payload") or {}
        ts = _pick_latest_timestamp(p) or ""
        amt = p.get("amount")
        m   = p.get("merchantName") or ""
        piece = _shorten(h.get("text") or m or "match")
        s = piece
        if amt is not None: s += f" · {_fmt_money(amt)}"
        if ts: s += f" · {ts}"
        view.append(s)
    return "\n".join(view)

# ------------------ legacy calculator fallbacks ------------------

def _render_legacy_value(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
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

# ------------------ RAG renderers ------------------

def _render_rag(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "I couldn't find anything relevant."
    ans = (payload.get("answer") or payload.get("result") or "").strip()
    if not ans:
        ans = "I couldn't find anything relevant."
    sources = payload.get("sources") or []
    if sources:
        src_lines = []
        for s in sources[:5]:
            src = s.get("source") or s.get("path") or s.get("file") or "source"
            snip = _shorten(s.get("snippet") or s.get("text") or "", 120)
            src_lines.append(f"- {src}: {snip}")
        ans += "\n\nSources:\n" + "\n".join(src_lines)
    return ans

# ------------------ public compose ------------------

def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> str:
    """
    Formats the heterogeneous `results` from execute_calls into a single answer string.
    """
    if not results:
        return "I couldn't find anything for that."

    lines: List[str] = []

    # Optional: include intent (debug)
    intent = (plan or {}).get("intent")
    if intent:
        lines.append(f"Intent: {intent}")

    # Consistent deterministic order: by key
    for key in sorted(results.keys()):
        payload = results.get(key)

        # key like "transactions.get_field[0]" -> op
        try:
            _, rest = key.split(".", 1)
            op = rest.split("[", 1)[0]
        except Exception:
            op = "value"

        if op == "get_field":
            val = payload.get("value")
            if val is None and isinstance(payload.get("row"), dict):
                fld = payload.get("field")
                r = payload["row"]
                # light fallback: try resolved_key, then raw field
                rk = payload.get("resolved_key")
                if rk and rk in r:
                    val = r.get(rk)
                elif fld and fld in r:
                    val = r.get(fld)
            lines.append(_render_get_field_payload(payload if isinstance(payload, dict) else {}))
        elif op == "find_latest":
            lines.append(_render_find_latest(payload if isinstance(payload, dict) else {}))
        elif op == "sum_where":
            lines.append(_render_sum_where(payload if isinstance(payload, dict) else {}))
        elif op == "topk_by_sum":
            lines.append(_render_topk_by_sum(payload if isinstance(payload, dict) else {}))
        elif op == "list_where":
            lines.append(_render_list_where(payload if isinstance(payload, dict) else {}))
        elif op == "semantic_search":
            lines.append(_render_semantic_search(payload if isinstance(payload, dict) else {}))
        elif op in ("unified_answer", "account_answer", "knowledge_answer"):
            lines.append(_render_rag(payload))
        else:
            # Legacy / unknown -> try legacy formatter, else dump short json
            rendered = _render_legacy_value(payload if isinstance(payload, dict) else {})
            if rendered:
                lines.append(rendered)
            else:
                try:
                    lines.append(_shorten(json.dumps(payload, ensure_ascii=False)))
                except Exception:
                    lines.append(str(payload))

    out = "\n\n".join(l for l in lines if l and str(l).strip())
    return out if out.strip() else "I couldn't find anything for that."