from datetime import datetime, timedelta, timezone
from __future__ import annotations
from typing import Any, Dict, List, Iterable, Optional, Tuple
from datetime import datetime, timedelta
import math
import re
import json

# --------------------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------------------

_DATE_KEYS = (
    "postedDateTime", "transactionDateTime", "paymentPostedDateTime",
    "paymentDateTime", "closingDateTime", "openingDateTime", "date"
)

_MONEY_FIELDS = {"amount", "interestCharged", "paymentAmount"}

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _now() -> datetime:
    # always timezone-aware
    return datetime.now(timezone.utc)

def _to_dt(v: Any) -> Optional[datetime]:
    """Parse ISO strings (with or without 'Z' / offset) into UTC tz-aware datetimes."""
    if not v:
        return None
    s = str(v).strip()
    if s.endswith("Z"):            # '...Z' → '+00:00' so fromisoformat can parse
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:          # naive → assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v:,.2f}"

def _ensure_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        # some loaders wrap rows
        if isinstance(data.get("rows"), list):
            return [r for r in data["rows"] if isinstance(r, dict)]
        # statements/transactions sometimes already list; else single object
        return [data]
    return []

# ---------------------- dotted path getter: 'persons[0].name' ----------------------
_BRACKETS = re.compile(r"\[(\d+)\]")

def _split_path(path: str) -> List[Any]:
    parts: List[Any] = []
    if not path:
        return parts
    for chunk in str(path).split("."):
        if not chunk:
            continue
        # pull out [idx] segments
        start = 0
        for m in _BRACKETS.finditer(chunk):
            key = chunk[start:m.start()]
            if key:
                parts.append(key)
            parts.append(int(m.group(1)))
            start = m.end()
        tail = chunk[start:]
        if tail:
            parts.append(tail)
    return parts

def _get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in _split_path(path):
        try:
            if isinstance(part, int) and isinstance(cur, list):
                cur = cur[part]
            elif isinstance(part, str) and isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return default
        except Exception:
            return default
    return cur if cur is not None else default

# ----------------------------- compute expressions -----------------------------
_ALLOWED_OPS = {"+", "-", "*", "/", "(", ")"}

def _compute_expr(expr: str, row: Dict[str, Any]) -> Optional[float]:
    """
    "__compute__(currentBalance/creditLimit)" -> evaluates using numeric
    fields from the row. Only + - * / and identifiers allowed.
    """
    m = re.match(r"__compute__\((.+)\)\s*$", str(expr or ""))
    if not m:
        return None
    src = m.group(1)
    # Replace identifiers with their numeric values if possible.
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\.\[\]]*|[\+\-\*\/\(\)]", src)
    safe: List[str] = []
    for t in tokens:
        if t in _ALLOWED_OPS:
            safe.append(t)
            continue
        if re.match(r"^[A-Za-z_]", t):
            val = _get_path(row, t)
            if val is None:
                return None
            try:
                val = float(val)
            except Exception:
                return None
            safe.append(str(val))
        else:
            return None
    expr_safe = " ".join(safe)
    try:
        return float(eval(expr_safe, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Filtering helpers
# --------------------------------------------------------------------------------------

_OP_MAP = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "=": "="}

def _amount_sign(row: Dict[str, Any]) -> Optional[str]:
    # DEBIT/CREDIT derived from amount when not explicitly present
    amt = row.get("amount")
    try:
        if amt is None:
            return None
        v = float(amt)
        if v > 0:
            return "DEBIT"
        if v < 0:
            return "CREDIT"
        return "ZERO"
    except Exception:
        return None

def _row_dt(row: Dict[str, Any]) -> Optional[datetime]:
    for k in _DATE_KEYS:
        dt = _to_dt(row.get(k))
        if dt:
            return dt
    return None

def _row_period(row: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit period if present ("YYYY-MM")
    p = row.get("period")
    if isinstance(p, str) and re.match(r"^\d{4}-(0[1-9]|1[0-2])$", p):
        return p
    dt = _row_dt(row)
    if not dt:
        return None
    return f"{dt.year:04d}-{dt.month:02d}"

def _match_simple(val: Any, cond: Any) -> bool:
    if isinstance(cond, dict):
        for op, rhs in cond.items():
            op = _OP_MAP.get(str(op).strip())
            if op is None:
                return False
            try:
                lv = float(val)
                rv = float(rhs)
            except Exception:
                # string compare for '=' only
                if op == "=":
                    if str(val).lower() != str(rhs).lower():
                        return False
                    continue
                return False
            if op == ">"  and not (lv >  rv): return False
            if op == ">=" and not (lv >= rv): return False
            if op == "<"  and not (lv <  rv): return False
            if op == "<=" and not (lv <= rv): return False
            if op == "="  and not (lv == rv): return False
        return True
    # literal equality (case-insensitive for strings)
    if isinstance(cond, str) and isinstance(val, str):
        return val.lower() == cond.lower()
    return val == cond

def _month_floor(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)

def _prev_month(dt: datetime, n: int = 1) -> datetime:
    y, m = dt.year, dt.month
    m -= n
    while m <= 0:
        m += 12
        y -= 1
    return datetime(y, m, 1, tzinfo=timezone.utc)

def _in_period_symbol(row: Dict[str, Any], sym: str) -> bool:
    dt = _row_dt(row)
    if not dt:
        return False
    sym = (sym or "").upper()
    today = _now()
    if sym == "THIS_YEAR":
        return dt.year == today.year
    if sym == "LAST_12M":
        return dt >= (today - timedelta(days=365))
    if sym == "LAST_MONTH":
        start = _month_floor(_prev_month(today, 1))
        end   = _month_floor(today)
        return start <= dt < end
    if sym == "PREV_MONTH":
        start = _month_floor(_prev_month(today, 2))
        end   = _month_floor(_prev_month(today, 1))
        return start <= dt < end
    m_q = re.match(r"^(\d{4})-Q([1-4])$", sym)
    if m_q:
        y = int(m_q.group(1));
        q = int(m_q.group(2))
        start_m = (q - 1) * 3 + 1
        start = datetime(y, start_m, 1, tzinfo=timezone.utc)
        end_m = start_m + 3
        end_y = y + (1 if end_m > 12 else 0)
        end_m = end_m if end_m <= 12 else end_m - 12
        end = datetime(end_y, end_m, 1, tzinfo=timezone.utc)
        return start <= dt < end

    m_m = re.match(r"^(\d{4})-(0[1-9]|1[0-2])$", sym)
    if m_m:
        y = int(m_m.group(1));
        m = int(m_m.group(2))
        start = datetime(y, m, 1, tzinfo=timezone.utc)
        end_m = m + 1;
        end_y = y + (1 if end_m > 12 else 0);
        end_m = 1 if end_m == 13 else end_m
        end = datetime(end_y, end_m, 1, tzinfo=timezone.utc)
        return start <= dt < end
    return False

def _match_where(row: Dict[str, Any], where: Dict[str, Any]) -> bool:
    if not where:
        return True
    for k, cond in where.items():
        if k == "period":
            # allow symbol or literal "YYYY-MM"
            if isinstance(cond, str):
                if not _in_period_symbol(row, cond):
                    return False
            elif isinstance(cond, dict):
                # support {"period": {"=": "YYYY-MM"}}
                if not _match_simple(_row_period(row), cond):
                    return False
            else:
                return False
            continue
        if k == "amountSign":
            sign = _amount_sign(row)
            if str(cond).upper() != str(sign).upper():
                return False
            continue
        val = _get_path(row, k) if isinstance(k, str) else None
        if not _match_simple(val, cond):
            return False
    return True

def _filter_rows(rows: List[Dict[str, Any]], where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if _match_where(r, where or {})]

def _sort_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    desc = sort_by.startswith("-")
    key  = sort_by[1:] if desc else sort_by
    def kf(r):
        v = _get_path(r, key)
        if key in _DATE_KEYS:
            dt = _to_dt(v)
            return dt or datetime.min.replace(tzinfo=timezone.utc)
        if isinstance(v, (int, float)): return v
        try: return float(v)
        except Exception: return str(v)
    return sorted(rows, key=kf, reverse=desc)

# --------------------------------------------------------------------------------------
# OPS: implementations
# --------------------------------------------------------------------------------------

def _op_get_field(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    row = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
    field = str(args.get("field") or "").strip()
    if not field:
        return {"error": "missing 'field'"}

    # compute expression
    if field.startswith("__compute__"):
        val = _compute_expr(field, row)
        return {"value": val, "row": row, "trace": {"field": field}} if val is not None else {"value": None, "row": row, "trace": {"field": field}}

    value = _get_path(row, field)
    return {"value": value, "row": row, "trace": {"field": field}}

def _op_find_latest(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows = _ensure_rows(data)
    where = args.get("where") or {}
    rows = _filter_rows(rows, where)
    if not rows:
        return {"value": None, "row": None, "trace": {"count": 0, "where": where}}

    key = str(args.get("field") or "postedDateTime")
    # sort by given key if present, else fallback to most plausible date
    if any(key in r for r in rows) or any(_get_path(r, key) is not None for r in rows):
        rows = _sort_rows(rows, f"-{key}")
    else:
        # fallback by first available date key
        for dk in _DATE_KEYS:
            if any(r.get(dk) for r in rows):
                rows = _sort_rows(rows, f"-{dk}")
                break

    top = rows[0]
    return {"value": _get_path(top, key), "row": top, "trace": {"count": len(rows), "where": where, "key": key}}

def _op_sum_where(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows = _ensure_rows(data)
    where = args.get("where") or {}
    vpath = str(args.get("value_path") or "amount")
    ret_avg = bool(args.get("return_avg", False))

    rows = _filter_rows(rows, where)
    total = 0.0; cnt = 0
    for r in rows:
        v = _get_path(r, vpath)
        try:
            total += float(v)
            cnt += 1
        except Exception:
            continue
    out: Dict[str, Any] = {"total": total, "count": cnt, "trace": {"where": where, "value_path": vpath}}
    if ret_avg:
        out["avg"] = (total / cnt) if cnt else 0.0
    return out

def _op_topk_by_sum(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows = _ensure_rows(data)
    where = args.get("where") or {}
    gkey  = str(args.get("group_key") or "merchantName")
    vpath = str(args.get("value_path") or "amount")
    k     = int(args.get("k", 5))

    rows = _filter_rows(rows, where)
    agg: Dict[str, float] = {}
    for r in rows:
        key = _get_path(r, gkey)
        if key is None:
            continue
        try:
            agg[str(key).strip()] = agg.get(str(key).strip(), 0.0) + float(_get_path(r, vpath) or 0.0)
        except Exception:
            continue
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return {"top": [{"key": k, "total": v} for k, v in top], "trace": {"where": where, "group_key": gkey, "value_path": vpath, "k": k}}

def _op_list_where(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows = _ensure_rows(data)
    where = dict(args.get("where") or {})

    # convenience filters
    min_amt = args.get("min_amount")
    max_amt = args.get("max_amount")
    merch_like = args.get("merchant_like")
    cat_like   = args.get("category_like")
    sort_by    = str(args.get("sort_by") or "").strip()
    limit      = int(args.get("limit", 50))
    offset     = int(args.get("offset", 0))

    flt = _filter_rows(rows, where)

    if min_amt is not None:
        try: flt = [r for r in flt if float(r.get("amount", 0)) >= float(min_amt)]
        except Exception: pass
    if max_amt is not None:
        try: flt = [r for r in flt if float(r.get("amount", 0)) <= float(max_amt)]
        except Exception: pass
    if merch_like:
        s = str(merch_like).lower()
        flt = [r for r in flt if s in str(r.get("merchantName", "")).lower()]
    if cat_like:
        s = str(cat_like).lower()
        flt = [r for r in flt if s in str(r.get("category", "")).lower()]

    if sort_by:
        flt = _sort_rows(flt, sort_by)

    items = flt[offset: offset + limit]
    out: Dict[str, Any] = {"items": items, "trace": {"where": where, "count": len(flt), "limit": limit, "offset": offset}}
    # aggregation block (optional)
    if isinstance(args.get("aggregate"), dict):
        agg_cfg = args["aggregate"]
        agg_res: Dict[str, int] = {}
        for name, sub_where in agg_cfg.items():
            agg_res[name] = sum(1 for r in flt if _match_where(r, sub_where or {}))
        out["aggregate"] = agg_res
    return out

def _op_semantic_search(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the domain plugin exposes a vector search method `search(query, k)`,
    we use it. Otherwise fallback to simple keyword matching on
    merchantName/description/category/memo.
    """
    query = str(args.get("query") or "").strip()
    k     = int(args.get("k", 12))
    must  = [str(x).lower() for x in (args.get("must_include") or [])]
    alts  = [str(x) for x in (args.get("alternates") or [])]

    # vector / plugin search hook
    if hasattr(plugin, "search") and callable(getattr(plugin, "search")):
        try:
            hits = plugin.search(query=query, k=k, alternates=alts, must_include=must)
            # Expected shape already hits; normalize minimal
            return {"hits": hits, "trace": {"mode": "plugin"}}
        except Exception:
            # fall back to keyword
            pass

    # simple keyword fallback
    rows = _ensure_rows(data)
    q_tokens = [t.lower() for t in _WORD_RE.findall(query)]
    fields = ("merchantName", "description", "memo", "category")
    scored: List[Tuple[float, Dict[str, Any]]] = []

    def text_of(r: Dict[str, Any]) -> str:
        parts = [str(r.get(f, "")) for f in fields]
        return " ".join(parts).lower()

    for r in rows:
        txt = text_of(r)
        if must and not all(m in txt for m in must):
            continue
        score = sum(txt.count(t) for t in q_tokens) + sum(txt.count(a.lower()) for a in alts)
        if score > 0:
            scored.append((float(score), r))

    scored.sort(key=lambda kv: kv[0], reverse=True)
    out_hits = []
    for _, r in scored[:k]:
        snippet = " ".join([str(r.get("merchantName") or ""), str(r.get("description") or "")]).strip()
        out_hits.append({
            "score": _,
            "text": snippet or "match",
            "payload": r
        })
    return {"hits": out_hits, "trace": {"mode": "keyword", "k": k}}

def _op_compare_periods(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows = _ensure_rows(data)
    p1 = str(args.get("period1") or "LAST_MONTH")
    p2 = str(args.get("period2") or "PREV_MONTH")
    # optional filter for purchases only, etc.
    where = args.get("where") or {}

    def total_for(sym: str) -> float:
        rs = [r for r in rows if _in_period_symbol(r, sym) and _match_where(r, where)]
        tot = 0.0
        for r in rs:
            try: tot += float(r.get("amount", 0))
            except Exception: pass
        return tot

    a = total_for(p1); b = total_for(p2)
    delta = a - b
    ratio = (a / b) if b not in (0, 0.0) else None
    return {"a_total": a, "b_total": b, "delta": delta, "ratio": ratio, "trace": {"period1": p1, "period2": p2, "where": where}}


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    return _to_dt(s)

def _in_range(ts: Optional[datetime], start: Optional[datetime], end: Optional[datetime]) -> bool:
    if ts is None:
        return False
    if start and ts < start:
        return False
    if end and ts > end:
        return False
    return True

def _as_rows(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list): return [r for r in x if isinstance(r, dict)]
    if isinstance(x, dict): return [x]
    return []

def op_explain_interest(*, domain: str, data: Any, args: Dict[str, Any], plugin: Any, scratch: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine latest statement (with interest) + in-period transactions (+ payments)
    to produce a grounded explanation.
    Returns a rich payload consumed by compose_answer.
    """
    st_rows = _as_rows(getattr(plugin, "rows", lambda: data)())
    if not st_rows:
        return {"error": "no_statements"}

    # 1) Pick statement row
    period_mode = (args or {}).get("period") or "LATEST_NONZERO"
    def _has_interest(r: Dict[str, Any]) -> bool:
        v = r.get("interestCharged")
        try: return (v or 0) != 0
        except Exception: return bool(v)

    st_rows_sorted = sorted(
        st_rows,
        key=lambda r: _parse_dt(r.get("closingDateTime")) or _parse_dt(r.get("date")) or datetime.min
    )
    if period_mode == "LATEST_NONZERO":
        st = next((r for r in reversed(st_rows_sorted) if _has_interest(r)), st_rows_sorted[-1])
    else:
        st = st_rows_sorted[-1]  # latest

    start = _parse_dt(st.get("openingDateTime") or st.get("periodStartDateTime"))
    end   = _parse_dt(st.get("closingDateTime") or st.get("periodEndDateTime"))
    st_interest = float(st.get("interestCharged") or 0.0)
    trailing    = float(st.get("totalTrailingInterest") or 0.0)
    non_trailing = max(0.0, st_interest - trailing)

    # 2) Load transactions (+ payments)
    load_domain = (ctx or {}).get("load_domain")
    tx_data = load_domain("transactions") if load_domain else None
    pay_data = load_domain("payments")     if load_domain else None

    tx_rows = _as_rows(tx_data) if tx_data else []
    pay_rows = _as_rows(pay_data) if pay_data else []

    # 3) In-period filters
    def _ts_tx(r: Dict[str, Any]) -> Optional[datetime]:
        return _parse_dt(
            r.get("postedDateTime") or r.get("transactionDateTime") or r.get("date")
        )
    tx_in_period = [r for r in tx_rows if _in_range(_ts_tx(r), start, end)]

    def _ts_pay(r: Dict[str, Any]) -> Optional[datetime]:
        return _parse_dt(r.get("paymentPostedDateTime") or r.get("paymentDateTime") or r.get("date"))
    pays_in_period = [r for r in pay_rows if _in_range(_ts_pay(r), start, end)]

    # 4) Summaries
    def _amt(r):
        try: return float(r.get("amount") or 0.0)
        except Exception: return 0.0

    # Purchases = positive spend; refunds negative (or vice-versa depending on file).
    # Use sign hints if present; otherwise use category/type to bucket.
    def _is_purchase(r):
        t = (r.get("transactionType") or "").upper()
        disp = (r.get("displayTransactionType") or "").lower()
        return ("PURCHASE" in t) or ("debit" in disp) or ("interest" not in disp)

    purchases_total = sum(abs(_amt(r)) for r in tx_in_period if _is_purchase(r))
    interest_txns   = [r for r in tx_in_period if "interest" in (r.get("displayTransactionType","").lower() or r.get("transactionType","").lower())]
    interest_tx_total = sum(abs(_amt(r)) for r in interest_txns)
    payments_total  = sum(abs(_amt(r)) for r in pays_in_period)

    carried_balance_hint = float(st.get("previousStatementBalance") or st.get("openingBalance") or 0.0)

    # 5) Build explanation
    return {
        "explain_interest": True,
        "period": {
            "start": st.get("openingDateTime") or st.get("periodStartDateTime"),
            "end":   st.get("closingDateTime") or st.get("periodEndDateTime"),
        },
        "statement": {
            "closingDateTime": st.get("closingDateTime"),
            "interestCharged": st_interest,
            "trailingInterest": trailing,
            "nonTrailingInterest": non_trailing,
            "isTrailingInterestApplied": bool(st.get("isTrailingInterestApplied", trailing > 0)),
        },
        "drivers": {
            "carried_balance_estimate": carried_balance_hint,
            "purchases_in_period": purchases_total,
            "payments_in_period": payments_total,
            "interest_transactions_total": interest_tx_total,
        },
        "support": {
            "statement_row": st,
            "interest_txn_ids": [r.get("transactionId") for r in interest_txns if r.get("transactionId")],
            "counts": {
                "tx_in_period": len(tx_in_period),
                "payments_in_period": len(pays_in_period),
            },
        },
        "trace": {"domain": domain, "op": "explain_interest"}
    }

# ---- register ----# --------------------------------------------------------------------------------------
# Public registry
# --------------------------------------------------------------------------------------

OPS = {
    "get_field": _op_get_field,
    "find_latest": _op_find_latest,
    "sum_where": _op_sum_where,
    "topk_by_sum": _op_topk_by_sum,
    "list_where": _op_list_where,
    "semantic_search": _op_semantic_search,
    "compare_periods": _op_compare_periods,
    "explain_interest": op_explain_interest,
}