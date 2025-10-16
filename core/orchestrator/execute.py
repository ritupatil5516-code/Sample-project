# core/orchestrator/execute.py
from __future__ import annotations
import yaml, json
from pathlib import Path
from typing import Any, Dict, List

from core.domains.registry import REGISTRY
from core.domains.base import OpContext

# session scratch (works even if you don't have runtime.py)
try:
    from src.api.contextApp.runtime import get_scratch
except Exception:
    _SCRATCH: Dict[str, Dict[str, Any]] = {}
    def get_scratch(session_id: str) -> Dict[str, Any]:
        box = _SCRATCH.get(session_id)
        if box is None:
            box = {}; _SCRATCH[session_id] = box
        return box

# RAG answerer (must be implemented in your repo)
try:
    from core.retrieval.rag_chain import unified_rag_answer
except Exception:
    def unified_rag_answer(question: str, session_id: str, account_id: str | None, cfg: Dict[str, Any], k: int = 6):
        return {"answer": "RAG not configured.", "sources": []}

_DOMAIN_ALIASES = {"account_summary": "accounts", "accounts": "accounts"}

def _norm_domain(d: str) -> str:
    d = (d or "").lower().replace("-", "_")
    return _DOMAIN_ALIASES.get(d, d)

def _read_app_cfg(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists(): return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _is_empty(res: Dict[str, Any]) -> bool:
    if not res: return True
    if "error" in res: return True
    if isinstance(res.get("items"), list): return len(res["items"]) == 0
    if isinstance(res.get("hits"), list): return len(res["hits"]) == 0
    if "value" in res: return (res["value"] is None) or (res["value"] == "")
    return False  # totals may be 0.0 legitimately

def execute_calls(calls: List[dict], context: dict) -> Dict[str, Any]:
    cfg = _read_app_cfg(context["app_yaml"])
    account_id = context.get("account_id")
    session_id = context.get("session_id") or "default"
    question   = context.get("question") or ""
    top_k      = int(context.get("top_k", 6))

    scratch = get_scratch(session_id)
    data_cache: Dict[str, Any] = {}
    out: Dict[str, Any] = {}

    for i, call in enumerate(calls or []):
        dom_id = _norm_domain(call.get("domain_id", ""))
        cap    = str(call.get("capability", "")).lower().replace("-", "_")
        args   = dict(call.get("args") or {})
        strat  = (call.get("strategy") or "auto").lower()
        allow_fallback = bool(call.get("allow_rag_fallback", True))
        key    = f"{dom_id}.{cap}[{i}]"

        dom = REGISTRY.get(dom_id)
        if not dom:
            out[key] = {"error": f"Unknown domain '{dom_id}'", "trace": {"strategy": strat}}
            continue

        fn = dom.ops.get(cap)
        if not fn:
            out[key] = {"error": f"Unknown capability '{cap}' for domain '{dom_id}'", "trace": {"strategy": strat}}
            continue

        # load once per domain
        if dom_id not in data_cache:
            data_cache[dom_id] = dom.load(account_id, cfg)

        scratch["_domain_obj"] = dom
        ctx = OpContext(account_id=account_id, session_id=session_id, cfg={**cfg, "question": question}, scratch=scratch, trace=True)

        def _do_det():
            try:
                return fn(data_cache[dom_id], args, ctx)
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}

        def _do_rag():
            try:
                r = unified_rag_answer(question=question, session_id=session_id, account_id=account_id, cfg=cfg, k=top_k)
                return {"answer": r.get("answer"), "sources": r.get("sources", []), "trace": {"strategy": "rag"}}
            except Exception as e:
                return {"error": f"RAG:{type(e).__name__}: {e}"}

        if strat == "rag":
            res = _do_rag()
        elif strat == "deterministic":
            res = _do_det()
            if _is_empty(res) and allow_fallback:
                res = {"fallback": _do_rag(), "trace": {"strategy": "deterministic→rag"}}
        else:  # auto
            res = _do_det()
            if _is_empty(res) and allow_fallback:
                res = {"fallback": _do_rag(), "trace": {"strategy": "auto→rag"}}

        out[key] = res

    scratch.pop("_domain_obj", None)
    return out