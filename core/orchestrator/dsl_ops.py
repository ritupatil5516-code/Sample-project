# core/orchestrator/dsl_ops.py
from __future__ import annotations
from typing import Any, Dict, List
from core.domains.base import OpContext, Domain

def _aliases(domain: Domain) -> Dict[str, List[str]]:
    return domain.aliases or {}

def _get_with_alias(domain: Domain, row: dict, key: str):
    if key in row: return row.get(key)
    for k in _aliases(domain).get(key, []):
        if k in row: return row.get(k)
    return None

def _s(x): return "" if x is None else str(x)

def _f(x, default=None):
    try: return float(x)
    except Exception: return default

def _match_one(domain: Domain, row: dict, k: str, v) -> bool:
    val = _get_with_alias(domain, row, k)
    if isinstance(v, dict):
        if "$contains" in v:
            return _s(v["$contains"]).lower() in _s(val).lower()
        if "$in" in v:
            cand = [(_s(x)).lower() for x in (v["$in"] or [])]
            return _s(val).lower() in cand
        num = _f(val, None)
        if num is None: return False
        if "$gte" in v and not (num >= _f(v["$gte"], num)): return False
        if "$gt"  in v and not (num >  _f(v["$gt"],  num)): return False
        if "$lte" in v and not (num <= _f(v["$lte"], num)): return False
        if "$lt"  in v and not (num <  _f(v["$lt"],  num)): return False
        return True
    if isinstance(v, str) and v.endswith("*"):
        return _s(val).startswith(v[:-1])
    if isinstance(v, str):
        return _s(val).lower() == v.lower()
    return val == v

def _matches(domain: Domain, row: dict, where: Dict[str, Any] | None) -> bool:
    where = where or {}
    if domain.id == "transactions" and not any(k in where for k in ("status", "transactionStatus")):
        if not _match_one(domain, row, "status", "POSTED"): return False
    for k, v in where.items():
        if not _match_one(domain, row, k, v): return False
    return True

def _ts(domain: Domain, row: dict) -> str:
    for k in ("postedDateTime","transactionDateTime","paymentPostedDateTime",
              "paymentDateTime","closingDateTime","date","period"):
        v = row.get(k) if k != "date" else _get_with_alias(domain, row, "date")
        if v: return _s(v)
    return ""

def _spend(row: dict) -> float:
    amt = _f(row.get("amount"), 0.0)
    t = (row.get("transactionType") or row.get("displayTransactionType") or "").upper()
    if t in {"CREDIT","REFUND","PAYMENT"} or amt < 0: return 0.0
    return max(0.0, amt)

# ---------------- ops ----------------

def op_get_field(data: Any, args: Dict[str, Any], ctx: OpContext) -> Dict[str, Any]:
    field = args.get("field")
    if isinstance(data, dict):
        return {"value": data.get(field), "trace": {"from": "dict"}}
    dom = ctx.scratch.get("_domain_obj")
    where = args.get("where") or {}
    rows = [r for r in (data or []) if _matches(dom, r, where)]
    if not rows: return {"value": None, "trace": {"count": 0, "where": where}}
    rows.sort(key=lambda r: _ts(dom, r), reverse=True)
    return {"value": _get_with_alias(dom, rows[0], field), "row": rows[0], "trace": {"count": len(rows), "where": where}}

def op_list_where(data: Any, args: Dict[str, Any], ctx: OpContext) -> Dict[str, Any]:
    dom = ctx.scratch.get("_domain_obj")
    where = args.get("where") or {}
    rows = [r for r in (data or []) if _matches(dom, r, where)]
    # follow-up support for transactions
    if dom.id == "transactions" and not where:
        last = ctx.scratch.get("last_top_merchants") or []
        if last:
            lows = [s.lower() for s in last]
            rows = [r for r in rows if _s(_get_with_alias(dom, r, "merchantName")).lower() in lows]
    per_key = args.get("per_key")
    per_n   = int(args.get("limit_per_key") or 1)
    rows.sort(key=lambda r: _ts(dom, r), reverse=True)
    if per_key:
        by, out = {}, []
        for r in rows:
            k = _s(_get_with_alias(dom, r, per_key))
            if len(by.setdefault(k, [])) < per_n:
                by[k].append(r); out.append(r)
        return {"items": out[: int(args.get("limit", 50))],
                "count": len(rows), "trace": {"where": where, "grouped_by": per_key, "limit_per_key": per_n}}
    return {"items": rows[: int(args.get("limit", 50))], "count": len(rows), "trace": {"where": where}}

def op_sum_where(data: Any, args: Dict[str, Any], ctx: OpContext) -> Dict[str, Any]:
    dom = ctx.scratch.get("_domain_obj")
    rows = [r for r in (data or []) if _matches(dom, r, args.get("where"))]
    value_path = (args.get("value_path") or "amount")
    total = 0.0
    for r in rows:
        total += _spend(r) if dom.id == "transactions" else _f(_get_with_alias(dom, r, value_path), 0.0)
    return {"total": round(total, 2), "count": len(rows), "trace": {"where": args.get("where") or {}, "value_path": value_path}}

def op_topk_by_sum(data: Any, args: Dict[str, Any], ctx: OpContext) -> Dict[str, Any]:
    dom = ctx.scratch.get("_domain_obj")
    key_field = args.get("key_field") or "merchantName"
    where = args.get("where") or {}
    k = int(args.get("k") or 5)
    agg = {}
    for r in (data or []):
        if not _matches(dom, r, where): continue
        key = _s(_get_with_alias(dom, r, key_field)) or "UNKNOWN"
        agg[key] = agg.get(key, 0.0) + (_spend(r) if dom.id=="transactions" else _f(_get_with_alias(dom, r, "amount"), 0.0))
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:k]
    if dom.id == "transactions":
        ctx.scratch["last_top_merchants"] = [m for m, _ in ranked]
    return {"top": [{"key": m, "total": v} for m, v in ranked], "trace": {"where": where, "k": k, "group_key": key_field}}

def op_find_latest(data: Any, args: Dict[str, Any], ctx: OpContext) -> Dict[str, Any]:
    dom = ctx.scratch.get("_domain_obj")
    where = args.get("where") or {}
    rows = [r for r in (data or []) if _matches(dom, r, where)]
    if not rows: return {"value": None, "trace": {"count": 0, "where": where}}
    rows.sort(key=lambda r: _ts(dom, r), reverse=True)
    field = args.get("field")
    return {"value": _get_with_alias(dom, rows[0], field) if field else rows[0],
            "row": rows[0], "trace": {"count": len(rows), "where": where}}

def op_semantic_search(data: Any, args: Dict[str, Any], ctx: OpContext) -> Dict[str, Any]:
    # Optional: your FAISS registry
    try:
        from core.index.faiss_registry import query_index, Embedder
    except Exception:
        return {"hits": [], "trace": {"error": "faiss_registry not available"}}
    q = (args.get("query") or "").strip()
    if not q: return {"hits": [], "trace": {"error": "query required"}}
    emb_cfg = (ctx.cfg.get("embeddings") or {})
    embedder = Embedder(
        provider=(emb_cfg.get("provider") or "openai"),
        model=emb_cfg.get("model") or emb_cfg.get("openai_model") or "text-embedding-3-large",
        api_key="", api_base=emb_cfg.get("api_base") or emb_cfg.get("openai_base_url"),
    )
    idx_dir = ((ctx.cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    hits = query_index("transactions", q, top_k=int(args.get("k", 5)), index_dir=idx_dir, embedder=embedder)
    return {"hits": hits, "trace": {"k": int(args.get("k", 5)), "query": q}}