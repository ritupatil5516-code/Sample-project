# core/orchestrator/compose_answer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

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
    for k in ("postedDateTime","transactionDateTime","paymentPostedDateTime",
              "paymentDateTime","closingDateTime","openingDateTime","date","period"):
        if k in row and row[k]: return _fmt_dt_iso(str(row[k]))
    return None

def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    return []

# -------- renderers --------

def _render_get_field(value: Any) -> str:
    if isinstance(value, (int, float)):
        return _fmt_money(value) if abs(float(value)) >= 1 else str(value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

def _render_find_latest(payload: Dict[str, Any]) -> str:
    val = payload.get("value")
    row = payload.get("row") or {}
    ts = _pick_latest_timestamp(row)
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
    cols_pref = ["postedDateTime","transactionDateTime","paymentPostedDateTime",
                 "merchantName","description","amount","transactionStatus","displayTransactionType",
                 "category","period"]
    sample = items[0]; cols = [c for c in cols_pref if c in sample][:4] or list(sample.keys())[:4]
    header = " | ".join(cols); sep = " | ".join("---" for _ in cols)
    lines = [header, sep]
    for r in items[:15]:
        def cell(c):
            v = r.get(c)
            if c in ("postedDateTime","transactionDateTime","paymentPostedDateTime"): return _fmt_dt_iso(v)
            return _fmt_money(v) if c=="amount" else _shorten(v)
        lines.append(" | ".join(cell(c) for c in cols))
    extra = len(items) - min(15, len(items))
    if extra > 0: lines.append(f"... and {extra} more")
    return "\n".join(lines)

def _render_legacy_value(payload: Dict[str, Any]) -> Optional[str]:
    if "interest_total" in payload: return _fmt_money(payload["interest_total"])
    if "item" in payload and isinstance(payload["item"], dict):
        r = payload["item"]; ts = _pick_latest_timestamp(r); amt = r.get("amount")
        m = r.get("merchantName") or r.get("description") or ""
        s_amt = f" for {_fmt_money(amt)}" if amt is not None else ""
        s_ts = f" on {ts}" if ts else ""; s_m = f" at {m}" if m else ""
        return f"Latest transaction{s_m}{s_amt}{s_ts}".strip()
    if "top_merchants" in payload:
        rows = payload["top_merchants"];
        if not rows: return "No results."
        return "\n".join(f"{i+1}. {r.get('merchant','UNKNOWN')}: {_fmt_money(r.get('total',0))}" for i, r in enumerate(rows))
    if "total" in payload and isinstance(payload["total"], (int, float)): return _fmt_money(payload["total"])
    if "items" in payload and isinstance(payload["items"], list): return _render_list_where({"items": payload["items"]})
    return None

# -------- public --------

def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> str:
    if not results: return "I couldn't find anything for that."
    lines: List[str] = []
    intent = (plan or {}).get("intent")
    if intent: lines.append(f"Intent: {intent}")

    for key, payload in results.items():
        try:
            domain, rest = key.split(".", 1)
        except ValueError:
            domain, rest = "general", key
        op = rest.split("[", 1)[0]

        # RAG direct/fallback
        if "answer" in payload and isinstance(payload["answer"], str):
            lines.append(payload["answer"])
            srcs = payload.get("sources") or (payload.get("fallback") or {}).get("sources") or []
            if srcs:
                lines.append("Sources:\n" + "\n".join(f"- {s.get('source')}" for s in srcs[:5]))
            continue
        if "fallback" in payload and isinstance(payload["fallback"], dict) and "answer" in payload["fallback"]:
            lines.append(payload["fallback"]["answer"])
            srcs = payload["fallback"].get("sources") or []
            if srcs:
                lines.append("Sources:\n" + "\n".join(f"- {s.get('source')}" for s in srcs[:5]))
            continue

        # DSL ops
        if op == "get_field":
            lines.append(_render_get_field(payload.get("value")))
        elif op == "find_latest":
            lines.append(_render_find_latest(payload))
        elif op == "sum_where":
            lines.append(_render_sum_where(payload))
        elif op == "topk_by_sum":
            lines.append(_render_topk_by_sum(payload))
        elif op == "list_where":
            lines.append(_render_list_where(payload))
        elif op == "semantic_search":
            hits = payload.get("hits") or []
            if not hits: lines.append("No relevant matches found.")
            else:
                view = []
                for h in hits[:5]:
                    p = h.get("payload") or {}
                    ts = _pick_latest_timestamp(p) or ""
                    amt = p.get("amount"); m = p.get("merchantName") or ""
                    piece = _shorten(h.get("text") or m or "match")
                    s = piece
                    if amt is not None: s += f" · {_fmt_money(amt)}"
                    if ts: s += f" · {ts}"
                    view.append(s)
                lines.append("\n".join(view))
        else:
            rendered = _render_legacy_value(payload if isinstance(payload, dict) else {})
            if rendered: lines.append(rendered)
            else: lines.append(_shorten(json.dumps(payload, ensure_ascii=False)))

    out = "\n\n".join(l for l in lines if l and str(l).strip())
    return out if out.strip() else "I couldn't find anything for that."