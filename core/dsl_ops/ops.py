# dsl_ops/ops.py
from __future__ import annotations
from typing import Any, Dict, List

def _get_path(row: Dict[str, Any], path: str, default=None):
    cur: Any = row
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _pick_ts(row: Dict[str, Any], ts_keys: List[str]) -> str:
    for k in ts_keys or []:
        v = row.get(k)
        if v:
            return str(v)
    # final fallback: any 'date' key we find
    return str(row.get("date", ""))

def _is_number(x: Any) -> bool:
    try:
        _ = float(x)
        return True
    except Exception:
        return False

def _match_cell(val: Any, cond: Any) -> bool:
    if isinstance(cond, dict):
        if "$contains" in cond:
            needle = str(cond["$contains"]).lower()
            return needle in str(val or "").lower()
        if "$in" in cond:
            arr = cond["$in"] or []
            return str(val) in [str(a) for a in arr]
        if "$eq" in cond:
            return str(val) == str(cond["$eq"])
        if "$ne" in cond:
            return str(val) != str(cond["$ne"])
        # numeric comparisons
        try:
            fv = float(val)
        except Exception:
            return False
        if "$gte" in cond and not (fv >= float(cond["$gte"])): return False
        if "$lte" in cond and not (fv <= float(cond["lte"])):  return False
        if "$gt"  in cond and not (fv >  float(cond["gt"])):   return False
        if "$lt"  in cond and not (fv <  float(cond["lt"])):   return False
        return True

    if isinstance(cond, str) and cond.endswith("*"):
        return str(val or "").lower().startswith(cond[:-1].lower())
    return str(val) == str(cond)

def _match_row(domain: str, row: Dict[str, Any], where: Dict[str, Any], plugin) -> bool:
    if not where:
        return True
    aliases = getattr(plugin, "ALIASES", {}) or {}
    for k, want in (where or {}).items():
        key = aliases.get(k, k)
        got = _get_path(row, key, None)
        if not _match_cell(got, want):
            return False
    return True


# -------------------- Public DSL ops --------------------

def get_field(*, domain: str, data: Any, args: Dict[str, Any], plugin, scratch: Dict[str, Any]) -> Dict[str, Any]:
    field = (args.get("field") or "").strip()
    if not field:
        return {"error": "field is required"}
    field = (getattr(plugin, "ALIASES", {}) or {}).get(field, field)

    if getattr(plugin, "is_list", False):
        rows: List[Dict[str, Any]] = data or []
        if not isinstance(rows, list) or not rows:
            return {"error": "no rows"}
        ts_keys = getattr(plugin, "timestamp_keys", []) or []
        latest = max(rows, key=lambda r: _pick_ts(r, ts_keys))
        return {"value": _get_path(latest, field), "row": latest}
    else:
        if not isinstance(data, dict):
            return {"error": "domain expects dict"}
        return {"value": _get_path(data, field)}

def _passes_where(row: Dict[str, Any], where: Dict[str, Any]) -> bool:
    if not where:
        return True
    for k, cond in (where or {}).items():
        v = row.get(k)
        if isinstance(cond, dict):
            # numeric comparisons: {">": 0}, {">=": 10}, {"=": 0}
            for op, rhs in cond.items():
                try:
                    fv = float(v) if v is not None else None
                    fr = float(rhs)
                except Exception:
                    return False
                if op == ">" and not (fv is not None and fv >  fr): return False
                if op == ">=" and not (fv is not None and fv >= fr): return False
                if op == "=" and not (fv == fr): return False
                if op == "<" and not (fv is not None and fv <  fr): return False
                if op == "<=" and not (fv is not None and fv <= fr): return False
        else:
            # equality / contains for strings
            if isinstance(cond, str):
                if str(cond).lower() not in str(v).lower():
                    return False
            else:
                if v != cond:
                    return False
    return True

def _sort_key(row: Dict[str, Any], field: str):
    # works for ISO datetime or plain numbers; fall back to raw
    val = row.get(field)
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        # prioritize ISO datetimes lexicographically
        return val
    return ""

def find_latest(*, domain: str, data: Any, args: Dict[str, Any], plugin, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = plugin.rows(data)
    where = args.get("where") or {}
    field = args.get("field") or plugin.default_timestamp

    filt = [r for r in rows if _passes_where(r, where)]
    if not filt:
        return {"value": None, "row": None, "trace": {"count": 0, "where": where}}

    row = max(filt, key=lambda r: _sort_key(r, field))
    return {
        "value": row.get(field),
        "row": row,
        "trace": {"count": len(filt), "where": where, "field": field}
    }

def sum_where(*, domain: str, data: Any, args: Dict[str, Any], plugin, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = data or []
    if not isinstance(rows, list):
        return {"error": "sum_where expects list domain"}
    where = args.get("where") or {}
    sum_field = (args.get("sum_field") or "amount")
    sum_field = (getattr(plugin, "ALIASES", {}) or {}).get(sum_field, sum_field)

    rows = [r for r in rows if _match_row(domain, r, where, plugin)]
    total = 0.0
    for r in rows:
        v = r.get(sum_field)
        if _is_number(v):
            total += float(v)
    return {"total": total, "count": len(rows), "trace": {"sum_field": sum_field, "where": where}}

def topk_by_sum(*, domain: str, data: Any, args: Dict[str, Any], plugin, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = data or []
    if not isinstance(rows, list):
        return {"error": "topk_by_sum expects list domain"}
    where = args.get("where") or {}
    key_field = (args.get("key_field") or "merchantName")
    sum_field = (args.get("sum_field") or "amount")
    aliases = (getattr(plugin, "ALIASES", {}) or {})
    key_field = aliases.get(key_field, key_field)
    sum_field = aliases.get(sum_field, sum_field)
    k = int(args.get("k", 5))

    rows = [r for r in rows if _match_row(domain, r, where, plugin)]
    buckets: Dict[str, float] = {}
    for r in rows:
        key = str(_get_path(r, key_field, "UNKNOWN"))
        v = r.get(sum_field)
        if _is_number(v):
            buckets[key] = buckets.get(key, 0.0) + float(v)

    top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:k]
    scratch["last_top_keys"] = [name for name, _ in top]
    scratch["last_top_key_field"] = key_field
    return {
        "top": [{"key": name, "total": total} for name, total in top],
        "trace": {"where": where, "group_by": key_field, "sum": sum_field, "k": k, "rows": len(rows)},
    }

def list_where(*, domain: str, data: Any, args: Dict[str, Any], plugin, scratch: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = data or []
    if not isinstance(rows, list):
        return {"error": "list_where expects list domain"}
    where = dict(args.get("where") or {})
    aliases = (getattr(plugin, "ALIASES", {}) or {})
    per_key = aliases.get(args.get("per_key",""), args.get("per_key",""))

    # follow-up: if no where and we have last_top_keys, scope to those
    last_keys = scratch.get("last_top_keys") if not where else None
    last_key_field = scratch.get("last_top_key_field")
    if last_keys and last_key_field:
        where[last_key_field] = {"$in": last_keys}

    rows = [r for r in rows if _match_row(domain, r, where, plugin)]
    # newest first
    ts_keys = getattr(plugin, "timestamp_keys", []) or []
    rows.sort(key=lambda r: _pick_ts(r, ts_keys), reverse=True)

    limit = int(args.get("limit", 100))
    limit_per_key = int(args.get("limit_per_key", 0))
    if per_key and limit_per_key > 0:
        seen: Dict[str, int] = {}
        out: List[Dict[str, Any]] = []
        for r in rows:
            key = str(_get_path(r, per_key, "UNKNOWN"))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= limit_per_key:
                out.append(r)
        rows = out

    return {"items": rows[:limit], "trace": {"where": where, "limit": limit, "per_key": per_key, "limit_per_key": limit_per_key}}