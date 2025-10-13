# core/orchestrator/compose_answer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

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

def _shorten(s: Any, n: int = 80) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def _pick_latest_timestamp(row: Dict[str, Any]) -> Optional[str]:
    for k in ("postedDateTime", "transactionDateTime", "paymentPostedDateTime",
              "paymentDateTime", "closingDateTime", "openingDateTime", "date"):
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
    if isinstance(payload, dict):
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
    cols_pref = ["postedDateTime", "transactionDateTime", "paymentPostedDateTime",
                 "merchantName", "description", "amount", "transactionStatus",
                 "displayTransactionType", "category", "period"]
    sample = items[0]
    cols = [c for c in cols_pref if c in sample][:4] or list(sample.keys())[:4]
    header = " | ".join(cols)
    sep = " | ".join("---" for _ in cols)
    lines = [header, sep]
    for r in items[:15]:
        line = " | ".join(
            _shorten(
                _fmt_money(r.get(c)) if c == "amount"
                else (r.get(c) if c not in ("postedDateTime","transactionDateTime","paymentPostedDateTime")
                      else _fmt_dt_iso(r.get(c)))
            )
            for c in cols
        )
        lines.append(line)
    extra = len(items) - min(15, len(items))
    if extra > 0:
        lines.append(f"... and {extra} more")
    return "\n".join(lines)

# ------------------ legacy calculator fallbacks ------------------

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

# ------------------ public compose ------------------

def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> str:
    """
    Formats results from execute_calls into a single string.
    Supports:
      - universal get_field (value/values)
      - legacy calculators
      - semantic_search
      - rag.{account_answer,knowledge_answer}
    """
    if not results:
        return "I couldn't find anything for that."

    lines: List[str] = []
    intent = (plan or {}).get("intent")
    if intent:
        lines.append(f"Intent: {intent}")

    for key, payload in results.items():
        try:
            domain, rest = key.split(".", 1)
        except ValueError:
            domain, rest = "general", key
        op = rest.split("[", 1)[0]

        if op == "get_field":
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
            hits = (payload or {}).get("hits") if isinstance(payload, dict) else []
            if not hits:
                lines.append("No relevant matches found.")
            else:
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
                lines.append("\n".join(view))
        elif op in ("account_answer", "knowledge_answer"):
            ans = (payload or {}).get("answer") if isinstance(payload, dict) else None
            srcs = (payload or {}).get("sources") if isinstance(payload, dict) else None
            part = (ans or "").strip() or "I couldn't find anything relevant."
            if srcs:
                part += "\n\nSources:\n" + "\n".join(
                    f"- {s.get('source')}: {_shorten(s.get('snippet'),100)}" for s in (srcs or [])[:5]
                )
            lines.append(part)
        else:
            rendered = _render_legacy_value(payload if isinstance(payload, dict) else {})
            if rendered:
                lines.append(rendered)
            else:
                lines.append(_shorten(json.dumps(payload, ensure_ascii=False)))

    out = "\n\n".join(l for l in lines if l and str(l).strip())
    return out if out.strip() else "I couldn't find anything for that."