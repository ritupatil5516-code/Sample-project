# dsl_ops.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Iterable, Optional
from datetime import datetime

# If you already have faiss_registry.query_index, keep using it.
# Otherwise you can stub semantic_search w/ simple keyword contains.

# ---------------- parse helpers ----------------
def _as_list(x: Any) -> List[Dict[str, Any]]:
    return x if isinstance(x, list) else ([x] if isinstance(x, dict) else [])

def _iso(s: str) -> Optional[datetime]:
    if not s: return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _latest_key(r: Dict[str, Any]) -> Tuple:
    keys = ["postedDateTime", "transactionDateTime", "paymentPostedDateTime", "closingDateTime", "date"]
    for k in keys:
        if r.get(k):
            dt = _iso(str(r[k]))
            if dt:
                return (dt,)
    return (datetime.min,)

def _match_where(row: Dict[str, Any], where: Dict[str, Any]) -> bool:
    if not where: return True
    for k, v in where.items():
        cur = row.get(k)
        if isinstance(v, str):
            if str(cur).lower().find(v.lower()) < 0:
                return False
        else:
            if cur != v:
                return False
    return True

# ---------------- ops ----------------
def get_field_value(domain_rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, Any]:
    """
    Return the value of a field from the *first* row for that domain.
    If the domain is account_summary, pass the single dict. For list domains,
    we look at the latest row by timestamp.
    Supports dotted paths: persons[0].ownershipType
    """
    def _get_path(obj: Any, path: str):
        if not isinstance(path, str):
            return None
        cur = obj
        # normalize brackets: a[0].b -> a.0.b
        path = path.replace("[", ".").replace("]", "")
        for part in [p for p in path.split(".") if p]:
            if isinstance(cur, list):
                try:
                    cur = cur[int(part)]
                except Exception:
                    return None
            elif isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur

    rows = _as_list(domain_rows)
    row = rows[0] if rows and not isinstance(rows[0], dict) else (max(rows, key=_latest_key) if rows else {})
    value = _get_path(row, field)
    return {"value": value, "row": row}

def find_latest_row(domain_rows: Iterable[Dict[str, Any]], where: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rows = [r for r in _as_list(domain_rows) if _match_where(r, where or {})]
    if not rows:
        return {"row": None, "count": 0}
    latest = max(rows, key=_latest_key)
    return {"row": latest, "count": len(rows)}

def sum_where(domain_rows: Iterable[Dict[str, Any]], where: Dict[str, Any] | None = None, field: str = "amount") -> Dict[str, Any]:
    rows = [r for r in _as_list(domain_rows) if _match_where(r, where or {})]
    total = sum(float(r.get(field) or 0.0) for r in rows)
    return {"total": total, "count": len(rows)}

def topk_by_sum(domain_rows: Iterable[Dict[str, Any]], key_field: str, k: int = 5, where: Dict[str, Any] | None = None, amount_field: str = "amount") -> Dict[str, Any]:
    rows = [r for r in _as_list(domain_rows) if _match_where(r, where or {})]
    agg: Dict[str, float] = {}
    for r in rows:
        kf = (r.get(key_field) or "UNKNOWN").strip() if isinstance(r.get(key_field), str) else str(r.get(key_field))
        agg[kf] = agg.get(kf, 0.0) + float(r.get(amount_field) or 0.0)
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return {"top": [{"key": k, "total": v} for k, v in top], "count": len(rows)}

def semantic_search(domain: str, query: str, *, index_dir: str, embedder, top_k: int = 5) -> Dict[str, Any]:
    """
    Thin shim that calls your existing faiss_registry.query_index.
    """
    from src.api.contextApp.index.faiss_registry import query_index
    hits = query_index(domain=domain, query=query, top_k=top_k, index_dir=index_dir, embedder=embedder)
    return {"hits": hits}