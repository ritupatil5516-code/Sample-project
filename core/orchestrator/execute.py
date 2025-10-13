# core/orchestrator/execute.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import os, json, yaml
import jmespath  # pip install jmespath

# ---- Keep your existing loaders (unchanged) ----
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

# (Optional) Old calculator fallback (only used if a legacy call isn't mapped)
from domains.transactions import calculator as txn_calc
from domains.payments import calculator as pay_calc
from domains.statements import calculator as stmt_calc
from domains.account_summary import calculator as acct_calc

# FAISS semantic search
from core.index.faiss_registry import query_index, Embedder


# =========================
# Config & embedder
# =========================
def _read_app_cfg(app_yaml_path: str) -> dict:
    try:
        with open(app_yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _build_embedder_from_cfg(cfg: dict) -> Embedder:
    emb = (cfg.get("embeddings") or {})
    provider = (emb.get("provider") or "openai").lower()
    model = emb.get("openai_model") or emb.get("model") or "text-embedding-3-large"
    api_base = emb.get("openai_base_url") or emb.get("api_base") or None
    api_key_env = emb.get("openai_api_key_env") or "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env, "")
    return Embedder(provider=provider, model=model, api_key=api_key, api_base=api_base)


# =========================
# Universal JSON ops (5)
# =========================
def _ensure_list(x: Any) -> List[dict]:
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    return []

def _to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    try: return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except: return None

def _get(obj: dict, path: str) -> Any:
    # JMESPath handles dot paths, arrays, filters, to_string(), contains(), etc.
    return jmespath.search(path, obj)

def _op_get_field(rows: List[dict], key_path: str) -> Dict[str, Any]:
    obj = rows[0] if rows else {}
    return {"value": _get(obj, key_path)}

def _op_find_latest(rows: List[dict], ts_field: str, value_path: str, where: Optional[str]=None) -> Dict[str, Any]:
    items = _ensure_list(rows)
    if where: items = jmespath.search(where, items) or []
    items = [r for r in items if _to_dt(_get(r, ts_field))]
    if not items: return {"value": None}
    items.sort(key=lambda r: _to_dt(_get(r, ts_field)) or datetime.min, reverse=True)
    return {"value": _get(items[0], value_path), "row": items[0]}

def _op_sum_where(rows: List[dict], value_path: str, where: Optional[str]=None) -> Dict[str, Any]:
    items = _ensure_list(rows)
    if where: items = jmespath.search(where, items) or []
    total = 0.0
    for r in items:
        v = _get(r, value_path)
        try: total += float(v or 0)
        except: pass
    return {"total": round(total, 2), "count": len(items)}

def _op_topk_by_sum(rows: List[dict], group_key: str, value_path: str,
                    where: Optional[str]=None, k:int=5) -> Dict[str, Any]:
    items = _ensure_list(rows)
    if where: items = jmespath.search(where, items) or []
    agg: Dict[str, float] = {}
    for r in items:
        key = _get(r, group_key) or "UNKNOWN"
        try:
            agg[str(key)] = agg.get(str(key), 0.0) + float(_get(r, value_path) or 0)
        except:
            pass
    ranked = sorted(
        ({"key": k_, "total": round(v, 2)} for k_, v in agg.items()),
        key=lambda x: x["total"], reverse=True
    )
    return {"top": ranked[:max(1, int(k))], "groups": len(agg)}

def _op_list_where(rows: List[dict], where: Optional[str]=None,
                   sort_by: Optional[str]=None, desc: bool=True, limit:int=20) -> Dict[str, Any]:
    items = _ensure_list(rows)
    if where: items = jmespath.search(where, items) or []
    if sort_by:
        try: items.sort(key=lambda r: _get(r, sort_by) or "", reverse=bool(desc))
        except: pass
    return {"items": items[:max(1, int(limit))], "count": len(items)}


# =========================
# Legacy → 5-op aliases (so your current planner still works)
# =========================
_ALIAS: Dict[str, Dict[str, Any]] = {
    # account_summary
    "current_balance": {"op":"get_field", "domain":"account_summary", "args":{"key_path":"currentBalance"}},
    "available_credit":{"op":"get_field", "domain":"account_summary", "args":{"key_path":"availableCreditAmount"}},
    # statements
    "total_interest":  {"op":"find_latest","domain":"statements", "args":{"ts_field":"closingDateTime","value_path":"interestCharged"}},
    # transactions
    "top_merchants":   {"op":"topk_by_sum","domain":"transactions", "args":{
        "group_key":"merchantName",
        "value_path":"amount",
        "where":"[?transactionStatus=='POSTED' && contains(to_string(displayTransactionType),'PURCHASE') && amount > `0`]",
        "k":5
    }},
    "list_over_threshold":{"op":"list_where","domain":"transactions", "args":{
        "where":"[?amount >= `THRESH` && transactionStatus=='POSTED']",
        "sort_by":"postedDateTime","desc":True,"limit":50
    }},
    "find_by_merchant":{"op":"list_where","domain":"transactions", "args":{
        "where":"[?contains(to_string(merchantName), `Q`) && transactionStatus=='POSTED']",
        "sort_by":"postedDateTime","desc":True,"limit":50
    }},
    "spend_in_period": {"op":"sum_where","domain":"transactions", "args":{
        "value_path":"amount",
        "where":"[?transactionStatus=='POSTED' && contains(to_string(displayTransactionType),'PURCHASE') && amount > `0`]"
    }},
    # passthrough
    "semantic_search": {"op":"semantic_search", "domain":"transactions", "args":{}},
}

def _map_calls_to_ops(calls: List[dict]) -> List[dict]:
    """Turn legacy planner `calls` into generic `ops` via _ALIAS."""
    ops: List[dict] = []
    for c in (calls or []):
        dom = str(c.get("domain_id","")).strip().lower().replace("-","_")
        cap = str(c.get("capability","")).strip().lower().replace("-","_")
        args = dict(c.get("args") or {})
        if cap in _ALIAS:
            spec = json.loads(json.dumps(_ALIAS[cap]))  # deep copy
            spec["domain"] = spec.get("domain") or dom
            # light templating:
            if "threshold" in args and "where" in spec["args"]:
                spec["args"]["where"] = spec["args"]["where"].replace("THRESH", str(args["threshold"]))
            if "merchant_query" in args and "where" in spec["args"]:
                spec["args"]["where"] = spec["args"]["where"].replace("Q", str(args["merchant_query"]))
            # caller overrides:
            for k, v in args.items():
                spec["args"][k] = v
            ops.append(spec)
        else:
            # soft fallback: if it already *is* one of the 5 ops, just reuse it
            if cap in {"get_field","find_latest","sum_where","topk_by_sum","list_where","semantic_search"}:
                ops.append({"op":cap, "domain": dom, "args": args})
            else:
                # last resort: keep a fallback record so nothing crashes
                ops.append({"op":"__fallback__", "domain": dom, "capability": cap, "args": args})
    return ops


# =========================
# Public: execute_calls
# =========================
def execute_calls(calls: List[dict], config_paths: dict) -> Dict[str, Any]:
    """
    Universal executor for the tiny DSL (and legacy calls):
      - Prefer explicit ops if provided via config_paths["_ops"].
      - Else map legacy `calls` -> 5 ops using _ALIAS.
      - Executes one of: get_field, find_latest, sum_where, topk_by_sum, list_where, semantic_search.
      - If an op is "__fallback__", use your old calculators (compat mode).
    Returns: dict keyed by "domain.op[index]" with each op's result payload.
    """
    # 0) Load data once
    txns = load_transactions("data/folder/transactions.json")
    pays = load_payments("data/folder/payments.json")
    stmts = load_statements("data/folder/statements.json")
    acct  = load_account_summary("data/folder/account_summary.json")
    acct_rows = acct if isinstance(acct, list) else ([acct] if isinstance(acct, dict) else [])

    dom_rows: Dict[str, List[dict]] = {
        "transactions": txns,
        "payments": pays,
        "statements": stmts,
        "account_summary": acct_rows,
    }

    # 1) Config & embedder once
    cfg = _read_app_cfg(config_paths["app_yaml"])
    index_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    embedder = _build_embedder_from_cfg(cfg)

    # 2) Resolve ops: prefer explicit ops; else map calls → ops
    ops: List[dict] = config_paths.get("_ops") or _map_calls_to_ops(calls)

    # 3) Run
    out: Dict[str, Any] = {}
    for i, spec in enumerate(ops):
        op   = spec.get("op")
        dom  = (spec.get("domain") or "").strip().lower()
        args = dict(spec.get("args") or {})
        key  = f"{dom}.{op}[{i}]"

        rows = dom_rows.get(dom) or []

        if op == "get_field":
            res = _op_get_field(rows, args["key_path"])

        elif op == "find_latest":
            res = _op_find_latest(rows, args["ts_field"], args["value_path"], args.get("where"))

        elif op == "sum_where":
            res = _op_sum_where(rows, args["value_path"], args.get("where"))

        elif op == "topk_by_sum":
            res = _op_topk_by_sum(rows, args["group_key"], args["value_path"], args.get("where"), int(args.get("k", 5)))

        elif op == "list_where":
            res = _op_list_where(rows, args.get("where"), args.get("sort_by"), bool(args.get("desc", True)), int(args.get("limit", 20)))

        elif op == "semantic_search":
            q = (args.get("query") or args.get("category") or "").strip()
            k = int(args.get("k", 5))
            res = {"hits": query_index(dom, q, top_k=k, index_dir=index_dir, embedder=embedder)} if q else {"hits": [], "error": "query is required"}

        elif op == "__fallback__":
            # Only invoked if a legacy capability wasn't in _ALIAS.
            cap = spec.get("capability")
            if dom == "transactions":
                if cap == "last_transaction":
                    last = max(txns, key=lambda t: (t.get("transactionDateTime") or t.get("postedDateTime") or t.get("date") or "")) if txns else None
                    res = {"item": last, "trace": {"count": len(txns)}}
                elif cap == "spend_in_period":
                    res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))
                elif cap == "purchases_in_cycle":
                    res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))
                elif cap == "max_amount":
                    res = txn_calc.max_amount(txns, args.get("account_id"), args.get("period"),
                                              args.get("period_start"), args.get("period_end"),
                                              args.get("category"), int(args.get("top", 1)))
                elif cap == "aggregate_by_category":
                    res = txn_calc.aggregate_by_category(txns, args.get("account_id"),
                                                         args.get("period"), args.get("period_start"), args.get("period_end"))
                elif cap == "average_per_month":
                    res = txn_calc.average_per_month(txns, args.get("account_id"), args.get("period"),
                                                     args.get("period_start"), args.get("period_end"),
                                                     bool(args.get("include_credits", False)))
                else:
                    res = {"error": f"Unknown legacy capability {cap}"}

            elif dom == "payments":
                if cap == "last_payment":
                    res = pay_calc.last_payment(pays, args.get("account_id"))
                elif cap == "total_credited_year":
                    yr = args.get("year")
                    res = pay_calc.total_credited_year(pays, args.get("account_id"), int(yr) if yr else None)
                elif cap == "payments_in_period":
                    res = pay_calc.payments_in_period(pays, args.get("account_id"), args.get("period"))
                else:
                    res = {"error": f"Unknown legacy capability {cap}"}

            elif dom == "statements":
                if cap == "interest_breakdown":
                    res = stmt_calc.interest_breakdown(stmts, args.get("account_id"), args.get("period"))
                elif cap == "trailing_interest":
                    res = stmt_calc.trailing_interest(stmts, args.get("account_id"), args.get("period"))
                else:
                    res = {"error": f"Unknown legacy capability {cap}"}

            elif dom == "account_summary":
                if cap == "current_balance":
                    res = acct_calc.current_balance(acct)
                elif cap == "available_credit":
                    res = acct_calc.available_credit(acct)
                else:
                    res = {"error": f"Unknown legacy capability {cap}"}
            else:
                res = {"error": f"Unknown legacy domain {dom}"}

        else:
            res = {"error": f"Unknown op {op}"}

        out[key] = res

    return out