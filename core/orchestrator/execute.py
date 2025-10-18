# core/orchestrator/execute.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import threading
import re

from domains.registry import normalize_domain, get_plugin, load_domain
from dsl_ops import OPS as DSL_OPS
from src.core.runtime import RUNTIME
from core.retrieval.rag_chain import unified_rag_answer

# --------------------------- per-session scratch (follow-ups) ---------------------------
_SCRATCH_LOCK = threading.Lock()
_SCRATCH: Dict[str, Dict[str, Any]] = {}

def _get_scratch(session_id: str) -> Dict[str, Any]:
    sid = session_id or "default"
    with _SCRATCH_LOCK:
        return _SCRATCH.setdefault(sid, {})

# --------------------------- small helpers ---------------------------
def _normalize_cap(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_")

def _is_empty_result(res: Optional[Dict[str, Any]]) -> bool:
    if not res or not isinstance(res, dict): return True
    if "error" in res: return True
    if res.get("value") in (None, "", []): return True
    if res.get("items") == []: return True
    if res.get("top") == []: return True
    if "total" in res and "count" in res:
        try:
            if int(res.get("count", 0)) == 0:
                return True
        except Exception:
            return True
    return False

# ---- parse "field>0" / "field>=123.45" into where dict ----
_FILTER_RE = re.compile(r"^\s*(?P<f>\w+)\s*(?P<op>>=|<=|>|<|=)\s*(?P<v>-?\d+(?:\.\d+)?)\s*$")

def _parse_filter(expr: str) -> Optional[Dict[str, Any]]:
    if not expr:
        return None
    m = _FILTER_RE.match(str(expr))
    if not m:
        return None
    f = m.group("f")
    op = m.group("op")
    v  = float(m.group("v"))
    return {f: {op: v}}

def _normalize_args(dom: str, cap: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make planner args uniform for DSL ops:
      - accept legacy 'filter: "field>0"' and convert to where dict
      - coerce common numeric params
      - provide safer defaults (e.g., field for find_latest if missing)
    """
    out = dict(args or {})

    # filter → where
    if "filter" in out and "where" not in out:
        parsed = _parse_filter(out.get("filter"))
        if parsed:
            out["where"] = parsed
        out.pop("filter", None)

    # ints
    for k in ("k", "limit", "offset", "top_k"):
        if k in out:
            try:
                out[k] = int(out[k])
            except Exception:
                out[k] = out[k]

    # default field for find_latest by domain
    if cap == "find_latest" and not out.get("field"):
        out["field"] = {
            "transactions": "postedDateTime",
            "payments":     "paymentPostedDateTime",
            "statements":   "closingDateTime",
            "accounts":     "date",
        }.get(dom, "date")

    return out

# --------------------------- main executor ---------------------------
def execute_calls(calls: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes planner calls with strategy-aware routing:

      strategy == "deterministic"  -> run DSL op
      strategy startswith "rag"    -> run unified RAG (account + knowledge)
      strategy == "auto"           -> try deterministic, if weak/empty -> RAG fallback
    """
    cfg        = RUNTIME.cfg or {}
    session_id = str(context.get("session_id", "default"))
    account_id = context.get("account_id")
    question   = context.get("question", "")
    top_k      = int(context.get("top_k", 6))

    allow_rag_fallback = bool((cfg.get("execution") or {}).get("allow_rag_fallback", True))

    results: Dict[str, Any] = {}
    domain_cache: Dict[str, Any] = {}
    scratch = _get_scratch(session_id)

    for i, call in enumerate(calls or []):
        dom      = normalize_domain(call.get("domain_id", ""))
        cap      = _normalize_cap(call.get("capability", ""))
        args     = _normalize_args(dom, cap, dict(call.get("args") or {}))
        strategy = (call.get("strategy") or "deterministic").strip().lower()

        key = f"{dom}.{cap}[{i}]"

        # --------------------- RAG lane (explicit) ---------------------
        if strategy.startswith("rag"):
            try:
                rag_res = unified_rag_answer(
                    question=question,
                    session_id=session_id,
                    account_id=account_id,
                    k=top_k,
                )
                results[f"rag.unified_answer[{i}]"] = rag_res
            except Exception as e:
                results[f"rag.error[{i}]"] = {"error": f"rag_error: {type(e).__name__}: {e}"}
            continue

        # --------------------- Deterministic lane (DSL ops) ---------------------
        plugin = get_plugin(dom)
        if not plugin:
            results[key] = {"error": f"unknown domain '{dom}'"}
            continue

        if dom not in domain_cache:
            try:
                domain_cache[dom] = load_domain(dom, account_id, cfg)
            except Exception as e:
                results[key] = {"error": f"load_error: {type(e).__name__}: {e}"}
                continue
        data = domain_cache[dom]

        op = DSL_OPS.get(cap)
        if not op:
            results[key] = {"error": f"unknown capability '{cap}'"}
            continue

        try:
            det_res = op(domain=dom, data=data, args=args, plugin=plugin, scratch=scratch)
        except Exception as e:
            det_res = {"error": f"op_error: {type(e).__name__}: {e}"}

        # --------------------- Auto fallback to RAG if needed ---------------------
        if strategy == "auto" and allow_rag_fallback and _is_empty_result(det_res):
            try:
                rag_res = unified_rag_answer(
                    question=question,
                    session_id=session_id,
                    account_id=account_id,
                    k=top_k,
                )
                results[key] = {"deterministic": det_res, "fallback": rag_res}
            except Exception as e:
                results[key] = {
                    "deterministic": det_res,
                    "fallback": {"error": f"rag_error: {type(e).__name__}: {e}"}
                }
        else:
            results[key] = det_res

    return results