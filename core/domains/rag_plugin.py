from __future__ import annotations
from typing import Any, Dict, Optional
from .base import Domain, OpContext

# use your existing chain
from core.retrieval.rag_chain import unified_rag_answer

def _load(account_id: Optional[str], cfg: Dict[str, Any]):
    # RAG doesn't need a preloaded dataset; retrievers read persisted indexes.
    return None

def op_unified_answer(_data: Any, args: Dict[str, Any], ctx: OpContext) -> Dict[str, Any]:
    """
    args: {"k": 6}  (optional)
    ctx.cfg should carry "question"; ctx.account_id/session_id come from server.
    """
    question   = (ctx.cfg or {}).get("question") or args.get("question") or ""
    account_id = ctx.account_id
    session_id = ctx.session_id
    k          = int(args.get("k", 6))

    res = unified_rag_answer(
        question=question,
        session_id=session_id,
        account_id=account_id,
        cfg=ctx.cfg,
        k=k,
    )
    # normalize shape
    return {
        "answer":  res.get("answer"),
        "sources": res.get("sources", []),
        "trace":   {"k": k, "account_id": account_id}
    }

OPS = {
    "unified_answer": op_unified_answer,
}

DOMAIN = Domain(
    id="rag",
    load=_load,
    ops=OPS,
    aliases={},   # none needed
)