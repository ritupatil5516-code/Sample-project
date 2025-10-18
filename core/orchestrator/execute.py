# core/orchestrator/execute.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import threading

# Single source of truth for domain metadata + loaders
from domains.registry import normalize_domain, get_plugin, load_domain

# Deterministic, domain-agnostic operators
from dsl_ops import OPS as DSL_OPS

# Shared runtime (LLM, cfg). Must be initialized at app startup.
from src.core.runtime import RUNTIME

# RAG lane (unified account + knowledge)
from core.retrieval.rag_chain import unified_rag_answer


# --------------------------------------------------------------------------------------
# Per-session scratch to enable follow-ups like:
# “Where did I spend most?” -> “Give dates for each spend”
# --------------------------------------------------------------------------------------
_SCRATCH_LOCK = threading.Lock()
_SCRATCH: Dict[str, Dict[str, Any]] = {}

def _get_scratch(session_id: str) -> Dict[str, Any]:
    sid = session_id or "default"
    with _SCRATCH_LOCK:
        return _SCRATCH.setdefault(sid, {})


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _normalize_cap(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_")

def _is_empty_result(res: Optional[Dict[str, Any]]) -> bool:
    if not res or not isinstance(res, dict):
        return True
    if "error" in res:
        return True
    if res.get("value") in (None, "", []):
        return True
    if res.get("items") == []:
        return True
    if res.get("top") == []:
        return True
    if "total" in res and "count" in res:
        try:
            cnt = int(res.get("count", 0))
        except Exception:
            cnt = 0
        if cnt == 0:
            return True
    return False


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------
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
        args     = dict(call.get("args") or {})
        strategy = (call.get("strategy") or "deterministic").strip().lower()
        key      = f"{dom}.{cap}[{i}]"

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
                results[key] = {"deterministic": det_res, "fallback": {"error": f"rag_error: {type(e).__name__}: {e}"}}
        else:
            results[key] = det_res

    return results