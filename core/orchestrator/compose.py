# core/orchestrator/compose_answer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

# ---------------- formatting helpers ----------------

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
        "postedDateTime",
        "transactionDateTime",
        "paymentPostedDateTime",
        "closingDateTime",
        "openingDateTime",
        "date",
    ):
        if k in row and row[k]:
            return _fmt_dt_iso(str(row[k]))
    return None

def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    return []

# ---------------- renderers for DSL ops ----------------

def _render_get_field(payload: Dict[str, Any]) -> str:
    val = payload.get("value")
    if isinstance(val, (int, float)):
        # Heuristic: treat ≥ 1 as currency for nicer display
        return _fmt_money(val) if abs(float(val)) >= 1 else str(val)
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def _render_find_latest(payload: Dict[str, Any]) -> str:
    row = payload.get("row") or {}
    ts = _pick_latest_timestamp(row)
    head = _render_get_field({"value": row}) if not row else ""
    # If caller put a 'value' inside payload (your custom use), show it
    if "value" in payload:
        v = _render_get_field({"value": payload.get("value")})
        return f"{v}" + (f" (as of {ts})" if ts else "")
    return f"Latest item" + (f" · {ts}" if ts else "")

def _render_sum_where(payload: Dict[str, Any]) -> str:
    total = payload.get("total")
    cnt = payload.get("count")
    t = _fmt_money(total) if isinstance(total, (int, float)) else str(total)
    return f"{t}" + (f" across {cnt} items" if isinstance(cnt, int) else "")

def _render_topk_by_sum(payload: Dict[str, Any]) -> str:
    rows = payload.get("top") or []
    if not rows:
        return "No results."
    lines = []
    for i, r in enumerate(rows, 1):
        key = r.get("key", "UNKNOWN")
        tot = r.get("total", 0)
        lines.append(f"{i}. {key}: {_fmt_money(tot)}")
    return "\n".join(lines)

def _render_list_where(payload: Dict[str, Any]) -> str:
    items = _as_list(payload.get("items"))
    if not items:
        return "No matching items."
    # choose 4 columns that are most useful for finance rows
    cols_pref = [
        "postedDateTime",
        "transactionDateTime",
        "paymentPostedDateTime",
        "merchantName",
        "description",
        "amount",
        "transactionStatus",
        "displayTransactionType",
        "category",
        "period",
    ]
    sample = items[0]
    cols = [c for c in cols_pref if c in sample][:4] or list(sample.keys())[:4]
    header = " | ".join(cols)
    sep = " | ".join("---" for _ in cols)
    lines = [header, sep]
    for r in items[:15]:
        def _cell(c):
            if c in ("postedDateTime", "transactionDateTime", "paymentPostedDateTime"):
                return _fmt_dt_iso(r.get(c))
            if c == "amount":
                return _fmt_money(r.get(c))
            return _shorten(r.get(c))
        lines.append(" | ".join(_cell(c) for c in cols))
    extra = len(items) - min(15, len(items))
    if extra > 0:
        lines.append(f"... and {extra} more")
    return "\n".join(lines)

def _render_semantic_search(payload: Dict[str, Any]) -> str:
    hits = payload.get("hits") or []
    if not hits:
        return "No relevant matches found."
    view = []
    for h in hits[:5]:
        p = h.get("payload") or {}
        ts = _pick_latest_timestamp(p) or ""
        amt = p.get("amount")
        m = p.get("merchantName") or p.get("description") or ""
        piece = _shorten(h.get("text") or m or "match")
        s = piece
        if amt is not None:
            s += f" · {_fmt_money(amt)}"
        if ts:
            s += f" · {ts}"
        view.append(s)
    return "\n".join(view)

# ---------------- renderers for RAG payloads ----------------

def _render_rag(payload: Dict[str, Any]) -> str:
    ans = payload.get("answer")
    if not ans:
        return "I couldn’t find an answer."
    srcs = payload.get("sources") or []
    lines = [str(ans)]
    if srcs:
        lines.append("\nSources:")
        for s in srcs[:4]:
            src = s.get("source") or ""
            snip = _shorten(s.get("snippet") or "", 140)
            lines.append(f"• {src} — {snip}")
    return "\n".join(lines)

# ---------------- public API ----------------

def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> str:
    """
    Formats heterogeneous `results` into a final answer string.
    Supports:
      - DSL ops: get_field, find_latest, sum_where, topk_by_sum, list_where, semantic_search
      - RAG: unified/account/knowledge answer shapes
    """
    if not results:
        return "I couldn't find anything for that."

    lines: List[str] = []
    intent = (plan or {}).get("intent")
    if intent:
        lines.append(f"Intent: {intent}")

    # Render each key’s payload
    for key, payload in results.items():
        try:
            domain, rest = key.split(".", 1)
        except ValueError:
            domain, rest = "general", key
        op = rest.split("[", 1)[0]

        if domain == "rag":
            lines.append(_render_rag(payload if isinstance(payload, dict) else {}))
            continue

        if op == "get_field":
            lines.append(_render_get_field(payload))
        elif op == "find_latest":
            lines.append(_render_find_latest(payload))
        elif op == "sum_where":
            lines.append(_render_sum_where(payload))
        elif op == "topk_by_sum":
            lines.append(_render_topk_by_sum(payload))
        elif op == "list_where":
            lines.append(_render_list_where(payload))
        elif op == "semantic_search":
            lines.append(_render_semantic_search(payload))
        else:
            # last resort: stringify
            try:
                lines.append(json.dumps(payload, ensure_ascii=False))
            except Exception:
                lines.append(str(payload))

    out = "\n\n".join(l for l in lines if str(l).strip())
    return out if out.strip() else "I couldn't find anything for that."