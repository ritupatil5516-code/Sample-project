# core/orchestrator/execute.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta, timezone

# deterministic loaders / calculators
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

from domains.transactions import calculator as txn_calc
from domains.payments import calculator as pay_calc
from domains.statements import calculator as stmt_calc
from domains.account_summary import calculator as acct_calc

# vector search for semantic ops (transactions)
from core.index.faiss_registry import query_index, Embedder

# RAG lane with conversational memory
from core.retrieval.rag_chain import (
    unified_rag_answer,     # account JSONs + knowledge
    account_rag_answer,     # account JSONs only (optional)
    knowledge_rag_answer,   # knowledge only (optional)
)
def _normalize_calls(calls: Any) -> List[Dict[str, Any]]:
    """
    Coerce plan['calls'] into a list[dict] with keys:
      - domain_id
      - capability
      - args (dict)
    Accepts items that are strings (JSON or op-name), dicts with alternate keys, etc.
    Silently drops unparseable items.
    """
    if calls is None:
        return []
    if isinstance(calls, dict):
        calls = [calls]
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(calls):
        # Allow raw strings: JSON, or a bare op like "unified_answer"
        if isinstance(c, str):
            s = c.strip()
            if not s:
                continue
            if s.startswith("{") and s.endswith("}"):
                try:
                    c = json.loads(s)
                except Exception:
                    continue
            else:
                # assume a rag op; default domain to 'rag'
                c = {"domain_id": "rag", "capability": s, "args": {}}

        if not isinstance(c, dict):
            continue

        # Tolerate alternate key names from planner
        dom = c.get("domain_id") or c.get("domain") or c.get("domain_key") or ""
        cap = c.get("capability") or c.get("op") or c.get("operation") or c.get("action") or ""
        args = c.get("args") or c.get("params") or {}

        # Final shape
        c = {
            "domain_id": str(dom).strip().lower().replace("-", "_"),
            "capability": str(cap).strip().lower().replace("-", "_"),
            "args": args if isinstance(args, dict) else {}
        }
        out.append(c)
    return out

# --- put near the top of the file ----
from typing import Any, Dict, List, Optional
import json, re

GENERIC_OPS = {"get_field", "find_latest", "sum_where", "topk_by_sum", "list_where", "semantic_search"}

# tiny json-path getter: supports dot, [index]
_PATH_TOKEN_RE = re.compile(r"\.|\[(\d+)\]")

def _json_get(obj: Any, path: str) -> Any:
    if not path:
        return obj
    cur = obj
    # tokenize "a.b[0].c"
    parts: List[str] = []
    buf = ""
    i = 0
    while i < len(path):
        if path[i] == "[":
            # flush token
            if buf:
                parts.append(buf)
                buf = ""
            # read [num]
            j = path.find("]", i)
            if j == -1:
                return None
            parts.append(path[i:j+1])  # like "[0]"
            i = j + 1
        elif path[i] == ".":
            if buf:
                parts.append(buf)
                buf = ""
            i += 1
        else:
            buf += path[i]
            i += 1
    if buf:
        parts.append(buf)

    for p in parts:
        if not p:
            continue
        if p.startswith("[") and p.endswith("]"):
            # index
            try:
                idx = int(p[1:-1])
                if isinstance(cur, list) and 0 <= idx < len(cur):
                    cur = cur[idx]
                else:
                    return None
            except Exception:
                return None
        else:
            # dict key (case-insensitive fallback)
            if isinstance(cur, dict):
                if p in cur:
                    cur = cur[p]
                else:
                    # case-insensitive / underscore fallback
                    pl = p.lower().replace(" ", "").replace("_", "")
                    found = None
                    for k in cur.keys():
                        if isinstance(k, str) and k.lower().replace(" ", "").replace("_", "") == pl:
                            found = k
                            break
                    cur = cur.get(found) if found is not None else None
            else:
                return None
    return cur

# aliases for convenience — expand as needed
FIELD_ALIASES = {
    "account_summary": {
        "status": ["accountStatus", "highestPriorityStatus", "balanceStatus"],
        "balance_status": ["balanceStatus"],
        "current_balance": ["currentBalance"],
        "available_credit": ["availableCredit", "availableCreditLimit"],
        "credit_limit": ["creditLimit"],
    },
    "transactions": {
        "amount": ["amount"],
        "posted": ["postedDateTime", "transactionDateTime", "date"],
        "merchant": ["merchantName", "description"],
    },
    "payments": {
        "posted": ["paymentPostedDateTime", "paymentDateTime", "date"],
        "amount": ["amount"],
    },
    "statements": {
        "period": ["period"],
        "closing": ["closingDateTime"],
        "interest": ["interestCharged"],
    },
}

def _first_non_none(vals: List[Any]) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None

def _op_get_field(dom: str, args: Dict[str, Any], ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Works across domains.
    - For account_summary (single dict) → returns the value directly.
    - For array domains (transactions/payments/statements) → by default returns the
      value from the *latest* item (by typical date fields). You can override with:
        args.item = "latest" | "first" | integer index
        args.path = dotted or bracket path (e.g. "merchantName", "persons[0].ownershipType")
    """
    field = (args.get("field") or "").strip()
    path  = (args.get("path")  or field or "").strip()
    item  = (args.get("item")  or "latest")

    if dom not in ds:
        return {"error": f"Unknown domain {dom}"}

    data = ds[dom]

    # account_summary is usually a single dict
    if dom == "account_summary" and isinstance(data, dict):
        # 1) direct path
        val = _json_get(data, path) if path else None
        if val is None and field:
            # 2) alias search
            aliases = FIELD_ALIASES.get(dom, {}).get(field.lower(), [])
            if aliases:
                val = _first_non_none([_json_get(data, p) for p in aliases])
        # 3) brute-force shallow case-insensitive match
        if val is None and field:
            fl = field.lower().replace("_", "")
            for k, v in data.items():
                if isinstance(k, str) and k.lower().replace("_", "") == fl:
                    val = v
                    break
        return {"value": val}

    # array domains
    if isinstance(data, list) and data:
        rows = data
        # pick item
        idx = None
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
            idx = int(item)
        elif str(item).lower() == "first":
            idx = 0
        else:  # latest (default)
            def _latest_key(r):
                return (
                    r.get("postedDateTime")
                    or r.get("transactionDateTime")
                    or r.get("paymentPostedDateTime")
                    or r.get("paymentDateTime")
                    or r.get("closingDateTime")
                    or r.get("date")
                    or ""
                )
            rows = sorted(rows, key=_latest_key, reverse=True)
            idx = 0

        if not (0 <= idx < len(rows)):
            return {"error": f"Item index out of range: {idx}", "count": len(rows)}

        row = rows[idx]
        val = _json_get(row, path) if path else None
        if val is None and field:
            aliases = FIELD_ALIASES.get(dom, {}).get(field.lower(), [])
            if aliases:
                val = _first_non_none([_json_get(row, p) for p in aliases])
            if val is None and field in row:
                val = row[field]

        return {"value": val, "row": row}

    return {"error": f"No data for domain {dom}"}

# ----------------- helpers -----------------

from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone

_TS_KEYS = (
    "postedDateTime", "transactionDateTime", "paymentPostedDateTime",
    "paymentDateTime", "closingDateTime", "openingDateTime", "date",
)


def _pick_latest_row(rows: List[dict]) -> Optional[dict]:
    def _ts(row: dict) -> Optional[datetime]:
        for k in _TS_KEYS:
            v = row.get(k)
            if isinstance(v, str) and v:
                try:
                    return datetime.fromisoformat(v.replace("Z", "+00:00"))
                except Exception:
                    continue
        return None
    best = None
    best_ts = None
    for r in rows:
        t = _ts(r)
        if t is not None and (best_ts is None or t > best_ts):
            best, best_ts = r, t
    return best or (rows[-1] if rows else None)

def _get_by_path(obj: Any, path: str) -> Any:
    """Dot-path extractor, e.g. 'accountStatus' or 'person.0.personName'."""
    if path is None or path == "":
        return obj
    cur: Any = obj
    for token in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(token)]
                continue
            except Exception:
                return None
        if isinstance(cur, dict):
            cur = cur.get(token)
            continue
        # cannot go deeper
        return None
    return cur

def _apply_where(rows: List[dict], where: Dict[str, Any]) -> List[dict]:
    """Very small filter: all conditions must match; supports dot-path keys."""
    if not where:
        return rows
    out = []
    for r in rows:
        ok = True
        for k, v in where.items():
            if _get_by_path(r, k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out

def _within_period(txn: Dict[str, Any], period: str | None) -> bool:
    if not period:
        return True
    if period.upper() == "LAST_12M":
        ts = (txn.get("transactionDateTime") or txn.get("postedDateTime") or txn.get("date"))
        if not ts:
            return False
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        return dt >= cutoff
    return True

def _get_field_from_row(row: Dict[str, Any], field: str) -> Any:
    """
    Supports dot paths and case-insensitive fallback:
      - "currentBalance"
      - "persons[0].accountActivity.rewardsEarned"
    """
    if not field:
        return None
    # try dotted/indexed path
    try:
        cur: Any = row
        for part in field.replace("]", "").split("."):
            if "[" in part:
                key, idx = part.split("[", 1)
                cur = cur.get(key, [])
                cur = cur[int(idx)]
            else:
                cur = cur.get(part)
            if cur is None:
                break
        if cur is not None:
            return cur
    except Exception:
        pass
    # fallback: case-insensitive leaf scan
    f_low = field.lower()
    for k, v in row.items():
        if k.lower() == f_low:
            return v
    return None

def _latest_key(row: Dict[str, Any]) -> str:
    for k in ("transactionDateTime","postedDateTime","paymentPostedDateTime","paymentDateTime","closingDateTime","openingDateTime","date"):
        v = row.get(k)
        if v: return str(v)
    return ""

def _read_app_cfg(path: str) -> Dict[str, Any]:
    import yaml, os
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _build_embedder_from_cfg(cfg: Dict[str, Any]) -> Embedder:
    emb = (cfg.get("embeddings") or {})
    provider = (emb.get("provider") or "openai").lower()
    model    = emb.get("openai_model") or emb.get("model") or "text-embedding-3-large"
    api_base = emb.get("openai_base_url") or emb.get("api_base") or None
    api_key  = __import__("os").getenv(emb.get("openai_api_key_env") or "OPENAI_API_KEY", "")
    return Embedder(provider=provider, model=model, api_key=api_key, api_base=api_base)

# --------------- generic 5-op DSL executors -----------------------------------
def _op_get_field(dom: str, args: Dict[str, Any], ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    get_field for any domain. Arguments:
      - field: dot-path inside the chosen row (required)
      - where: {path: value} exact filters (optional)
      - latest: bool (optional, default False) → if True and domain has many rows,
                choose the latest by timestamp; else choose the first row.
    """
    field = (args.get("field") or "").strip()
    if not field:
        return {"error": "get_field requires 'field' path"}

    latest = bool(args.get("latest", False))
    where  = dict(args.get("where") or {})

    # pick domain rows
    if dom == "transactions":
        rows = list(ds.get("transactions") or [])
    elif dom == "payments":
        rows = list(ds.get("payments") or [])
    elif dom == "statements":
        rows = list(ds.get("statements") or [])
    elif dom == "account_summary":
        # account_summary could be a single dict or list
        val = ds.get("account_summary")
        rows = val if isinstance(val, list) else ([val] if isinstance(val, dict) else [])
    else:
        return {"error": f"Unknown domain for get_field: {dom}"}

    rows = _apply_where(rows, where)

    if not rows:
        return {"error": "No matching rows"}

    row = _pick_latest_row(rows) if latest else rows[0]
    value = _get_by_path(row, field)
    return {"value": value, "row": row}

def _op_find_latest(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    data = datasets.get(domain)
    if isinstance(data, dict):
        return {"value": data, "row": data}
    rows = data or []
    if not rows:
        return {"value": None}
    row = max(rows, key=_latest_key)
    return {"value": row, "row": row}

def _op_sum_where(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    field = (args.get("field") or "amount").strip()
    where = args.get("where") or {}
    rows: List[Dict[str, Any]] = datasets.get(domain) or []
    total = 0.0; cnt = 0
    for r in rows:
        ok = True
        for k, v in where.items():
            if r.get(k) != v:
                ok = False; break
        if ok:
            try:
                total += float(r.get(field) or 0)
                cnt += 1
            except Exception:
                pass
    return {"total": total, "count": cnt}

def _op_topk_by_sum(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    group_key = (args.get("group_by") or "merchantName").strip()
    field = (args.get("field") or "amount").strip()
    k = int(args.get("k", 5))
    rows: List[Dict[str, Any]] = datasets.get(domain) or []
    agg: Dict[str, float] = {}
    for r in rows:
        g = (r.get(group_key) or "UNKNOWN")
        try:
            agg[g] = agg.get(g, 0.0) + float(r.get(field) or 0)
        except Exception:
            pass
    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return {"top": [{"key": a, "total": b} for a, b in top]}

def _op_list_where(domain: str, args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    where = args.get("where") or {}
    period = args.get("period")
    rows: List[Dict[str, Any]] = datasets.get(domain) or []
    out = []
    for r in rows:
        if period and not _within_period(r, period):
            continue
        ok = True
        for k, v in where.items():
            if r.get(k) != v:
                ok = False; break
        if ok:
            out.append(r)
    out.sort(key=_latest_key, reverse=True)
    return {"items": out}

def _op_semantic_search_transactions(args: Dict[str, Any], index_dir: str, embedder: Embedder) -> Dict[str, Any]:
    q_raw = (args.get("query") or args.get("category") or "").strip()
    alts = [a for a in (args.get("alternates") or []) if isinstance(a, str) and a.strip()]
    k = int(args.get("k", 5))
    if not q_raw:
        return {"hits": [], "error": "query is required", "trace": {"k": k}}
    merged: Dict[str, Dict[str, Any]] = {}
    for q in [q_raw] + alts[:7]:
        for h in query_index("transactions", q, top_k=max(1, k), index_dir=index_dir, embedder=embedder):
            p = h.get("payload") or {}
            rid = p.get("transactionId") or h.get("idx")
            score = float(h.get("score", 0.0))
            prev = merged.get(rid)
            if (prev is None) or (score > prev["score"]):
                merged[rid] = {"score": score, "text": h.get("text"), "payload": p, "matched_query": q}
    hits = list(merged.values())
    hits.sort(key=lambda x: x["score"], reverse=True)
    return {"hits": hits[:k], "trace": {"k": k, "query": q_raw, "alternates": alts}}


# put near the top of execute.py if not already imported
from core.index.faiss_registry import query_index

def _row_id(dom: str, p: Dict[str, Any], h: Dict[str, Any]) -> str:
    """Best-effort record id for de-dup across alternates."""
    return (
        p.get("transactionId")
        or p.get("paymentId")
        or p.get("statementId")
        or p.get("id")
        or str(h.get("idx"))
    )

def _passes_must_include(h: Dict[str, Any], tokens: List[str]) -> bool:
    if not tokens:
        return True
    p = h.get("payload") or {}
    blob = " ".join(
        str(x)
        for x in [
            h.get("text", ""),
            p.get("merchantName", ""),
            p.get("description", ""),
            p.get("displayTransactionType", ""),
            p.get("category", ""),
            p.get("period", ""),
        ]
        if x
    ).lower()
    return all(t.strip().lower() in blob for t in tokens if isinstance(t, str) and t.strip())

def _op_semantic_search(
    dom: str,
    args: Dict[str, Any],
    ds: Dict[str, Any],
    index_dir: str,
    embedder,          # whatever you built earlier via your config (Embedder/OpenAI)
) -> Dict[str, Any]:
    """
    Embedding search against the FAISS index for the given domain.
    Args supported:
      - query: str (required)
      - alternates: List[str] (optional)
      - must_include: List[str] (optional lowercase tokens that MUST appear)
      - account_id: str (optional; filters payload.accountId)
      - k: int (top-K to return; default 6)
    """
    q_raw = (args.get("query") or args.get("category") or args.get("merchant_query") or "").strip()
    if not q_raw:
        return {"hits": [], "error": "query is required", "trace": {"domain": dom}}

    alts = [a for a in (args.get("alternates") or []) if isinstance(a, str) and a.strip()]
    must = [t for t in (args.get("must_include") or []) if isinstance(t, str) and t.strip()]
    k = max(1, int(args.get("k", 6)))
    aid = (args.get("account_id") or "").strip()

    merged: Dict[str, Dict[str, Any]] = {}
    queries = [q_raw] + alts[:8]  # cap alternates to keep it cheap

    for q in queries:
        try:
            hits = query_index(
                domain=dom,
                query=q,
                top_k=k,
                index_dir=index_dir,
                embedder=embedder,
            )
        except FileNotFoundError as e:
            return {"hits": [], "error": f"Index not found for domain '{dom}': {e}", "trace": {"domain": dom}}
        except Exception as e:
            return {"hits": [], "error": f"semantic_search failed: {e}", "trace": {"domain": dom}}

        for h in hits or []:
            p = h.get("payload") or {}
            # optional account filter
            if aid and p.get("accountId") != aid:
                continue
            # must_include tokens
            if not _passes_must_include(h, must):
                continue

            rid = _row_id(dom, p, h)
            score = float(h.get("score", 0.0))
            prev = merged.get(rid)
            if (prev is None) or (score > prev.get("score", 0.0)):
                merged[rid] = {
                    "score": score,
                    "text": h.get("text"),
                    "payload": p,
                    "matched_query": q,
                }

    out = list(merged.values())
    out.sort(key=lambda x: x["score"], reverse=True)
    return {
        "hits": out[:k],
        "trace": {
            "domain": dom,
            "query": q_raw,
            "alternates": alts,
            "must_include": must,
            "filtered_by_account": bool(aid),
            "returned": min(k, len(out)),
        },
    }

# dotted / bracket path getter that also supports list/tuple parts
def _get_path(obj, path):
    """
    Examples accepted:
      - "persons[0].ownershipType"
      - "persons.0.ownershipType"
      - ["persons", 0, "ownershipType"]
    """
    # Normalize to a list of parts
    if isinstance(path, (list, tuple)):
        parts = list(path)
    else:
        s = str(path)
        # turn bracket indexing into dotted segments -> persons[0] -> persons.0
        s = s.replace('[', '.').replace(']', '')
        parts = [p for p in s.split('.') if p]

    cur = obj
    for part in parts:
        # If current node is a list, treat part as an index
        if isinstance(cur, list):
            try:
                idx = part if isinstance(part, int) else int(str(part))
                cur = cur[idx]
            except Exception:
                return None
        # If current node is a dict, treat part as a key
        elif isinstance(cur, dict):
            key = part if isinstance(part, str) else str(part)
            cur = cur.get(key)
        else:
            return None

        if cur is None:
            return None

    return cur

# ----------------------------- main executor ----------------------------------
def execute_calls(calls: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg expects:
      - app_yaml: path to app.yaml (embeddings, index dir)
      - intent: optional string
      - question: original user text (for RAG lane)
      - session_id: conversation id for memory
    """
    results: Dict[str, Any] = {}
    calls = _normalize_calls(calls)
    # config for embeddings / FAISS
    app_yaml = (cfg or {}).get("app_yaml") or "config/app.yaml"
    app_cfg = _read_app_cfg(app_yaml)
    index_dir = ((app_cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    embedder = _build_embedder_from_cfg(app_cfg)

    # handy values for RAG
    session_id = (cfg or {}).get("session_id") or "default"
    question   = (cfg or {}).get("question") or ""

    # we’ll lazy-load datasets per account_id when needed
    cache: Dict[str, Dict[str, Any]] = {}  # account_id -> {domain->data}

    def _ensure_datasets(aid: str) -> Dict[str, Any]:
        ds = cache.get(aid)
        if ds is None:
            ds = {
                "transactions": load_transactions(aid),
                "payments":     load_payments(aid),
                "statements":   load_statements(aid),
                "account_summary": load_account_summary(aid),
            }
            cache[aid] = ds
        return ds

    for i, call in enumerate(calls or []):
        # 1) normalize the call safely
        if isinstance(call, str):
            try:
                call = json.loads(call)
            except Exception:
                results[f"call[{i}]"] = {"error": "malformed call"}
                continue

        dom = str(call.get("domain_id", "")).strip().lower().replace("-", "_")
        cap = str(call.get("capability", "")).strip().lower().replace("-", "_")
        args = dict(call.get("args") or {})
        key = f"{dom}.{cap}[{i}]"

        # optional account_id (used by ensure_datasets)
        account_id = args.get("account_id") or (cfg or {}).get("account_id")

        # 2) ---- Generic 5-op DSL FIRST -----------------------------------------
        if cap in GENERIC_OPS:
            # load datasets (transactions/payments/statements/account_summary)
            ds = _ensure_datasets(account_id) if account_id else _ensure_datasets(None)

            if cap == "get_field":
                ds = _ensure_datasets(account_id) if account_id else _ensure_datasets(None)

                rows = ds.get(dom)
                if rows is None:
                    results[key] = {"error": f"No data for domain '{dom}'"}
                    continue

                # support single-object JSON (e.g., account_summary)
                if isinstance(rows, dict):
                    rows_list = [rows]
                elif isinstance(rows, list):
                    rows_list = rows
                else:
                    results[key] = {"error": f"Unsupported data type for domain '{dom}'"}
                    continue

                # optional account filter if accountId exists in the row
                if account_id:
                    filtered = [r for r in rows_list if
                                str(r.get("accountId") or r.get("account_id")) == str(account_id)]
                    rows_list = filtered or rows_list

                if not rows_list:
                    results[key] = {"error": "No matching rows"}
                    continue

                row = rows_list[0]

                field = (args.get("field") or "").strip()
                if not field:
                    results[key] = {"error": "field is required"}
                    continue

                # alias resolution
                field_map = FIELD_ALIASES.get(dom, {})
                key_name = field_map.get(field, field)

                # dotted-path getter (supports indices like persons[0].ownershipType)
                def _get_path(d: dict, path: str):
                    cur = d
                    for part in path.replace("[", ".").replace("]", "").split("."):
                        if not part:
                            continue
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

                value = _get_path(row, key_name)
                if value is None and key_name in row:  # fallback if not dotted
                    value = row[key_name]

                if value is None and field != key_name and field in row:
                    # final safety: if alias didn't hit but the raw field exists
                    value = row[field]

                if value is None:
                    results[key] = {"value": None, "row": row, "field": field, "resolved_key": key_name,
                                    "error": f"Field '{field}' not found"}
                else:
                    results[key] = {"value": value, "row": row, "field": field, "resolved_key": key_name}
                continue
            elif cap == "find_latest":
                res = _op_find_latest(dom, args, ds)  # implement similar helpers as needed
            elif cap == "sum_where":
                res = _op_sum_where(dom, args, ds)
            elif cap == "topk_by_sum":
                res = _op_topk_by_sum(dom, args, ds)
            elif cap == "list_where":
                res = _op_list_where(dom, args, ds)
            elif cap == "semantic_search":
                res = _op_semantic_search(dom, args, ds, index_dir, embedder)
            else:
                res = {"error": f"Unsupported op {cap}"}

            results[key] = res
            continue  # IMPORTANT: do not fall through

        # 3) ---- RAG lane (explicit) ---------------------------------------------
        if dom == "rag":
            k = int(args.get("k", 6))
            try:
                if cap in ("unified_answer", "unified"):
                    if not account_id:
                        res = {"error": "account_id is required for rag.unified_answer"}
                    else:
                        res = unified_rag_answer(
                            question=question,
                            session_id=session_id,
                            account_id=account_id,
                            k=k,
                        )
                elif cap in ("account_answer", "account_only"):
                    if not account_id:
                        res = {"error": "account_id is required for rag.account_answer"}
                    else:
                        res = account_rag_answer(
                            question=question,
                            session_id=session_id,
                            account_id=account_id,
                            k=k,
                        )
                elif cap in ("knowledge_answer", "policy_answer", "handbook_answer"):
                    res = knowledge_rag_answer(
                        question=question,
                        session_id=session_id,
                        k=k,
                    )
                else:
                    res = {"error": f"Unknown rag capability '{cap}'"}
            except FileNotFoundError as e:
                res = {"error": f"Index not found: {e}"}
            except Exception as e:
                res = {"error": f"RAG execution error: {e}"}

            results[key] = res
            continue

        # 4) ---- If you still want legacy calculators, put them here -------------
        # elif dom == "transactions": ... etc.

        # 5) ---- Unknown ----------------------------------------------------------
        results[key] = {"error": f"Unknown capability {cap} for domain {dom}"}

    return results