# core/retrieval/rag_chain.py
from __future__ import annotations
from typing import Dict, Any, List

from langchain.chains import ConversationalRetrievalChain
from core.providers import read_cfg, build_langchain_llm
from core.memory.memory_store import MEMORY

# LlamaIndex-based indices/retrievers
from core.retrieval.llama_indices import (
    ensure_account_index,
    ensure_knowledge_index,
)

# -------------------- helpers --------------------

def _llm_from_cfg():
    cfg = read_cfg("config/app.yaml")
    return build_langchain_llm(cfg, temperature=0.0)

def _crc(llm, retriever, session_id: str):
    mem = MEMORY.get(session_id)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=mem,
        return_source_documents=True
    )

def _fmt_sources(docs: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for d in docs[:5]:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("file_path") or meta.get("path") or meta.get("domain") or "source"
        txt = getattr(d, "page_content", "") or ""
        out.append({"source": str(src), "snippet": txt[:180]})
    return out

def _knowledge_paths_from_cfg() -> List[str]:
    cfg = read_cfg("config/app.yaml")
    kn = (cfg.get("knowledge") or {})
    paths = kn.get("paths") or []
    if paths:
        return [str(p) for p in paths]
    # sensible defaults if not configured
    return [
        "data/knowledge/handbook.md",
        "data/agreement/Apple-Card-Customer-Agreement.pdf",
    ]

# -------------------- public API --------------------

def unified_rag_answer(
    question: str,
    session_id: str,
    account_id: str,
    txns: List[Dict[str, Any]],
    pays: List[Dict[str, Any]],
    stmts: List[Dict[str, Any]],
    acct: Dict[str, Any],
    knowledge_paths: List[str] | None = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    """
    Combines account retriever (transactions/payments/statements/account_summary)
    with global knowledge (handbook/policy) using an ensemble retriever.
    """
    llm = _llm_from_cfg()

    # Build / load indices
    acc_index = ensure_account_index(account_id, txns, pays, stmts, acct)
    kn_index  = ensure_knowledge_index(knowledge_paths or _knowledge_paths_from_cfg())

    # Convert to LangChain retrievers
    r1 = acc_index.as_langchain_retriever(similarity_top_k=top_k)
    r2 = kn_index.as_langchain_retriever(similarity_top_k=top_k)

    # Simple ensemble: weight account higher than knowledge
    from langchain.retrievers import EnsembleRetriever
    retriever = EnsembleRetriever(retrievers=[r1, r2], weights=[0.7, 0.3])

    chain = _crc(llm, retriever, session_id)
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    return {
        "answer": out.get("answer") or out.get("result"),
        "sources": _fmt_sources(docs),
    }

def account_rag_answer(
    question: str,
    session_id: str,
    account_id: str,
    txns: List[Dict[str, Any]],
    pays: List[Dict[str, Any]],
    stmts: List[Dict[str, Any]],
    acct: Dict[str, Any],
    top_k: int = 6,
) -> Dict[str, Any]:
    """
    RAG over the *account’s* four JSON datasets only.
    """
    llm = _llm_from_cfg()
    acc_index = ensure_account_index(account_id, txns, pays, stmts, acct)
    retriever = acc_index.as_langchain_retriever(similarity_top_k=top_k)

    chain = _crc(llm, retriever, session_id)
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    return {
        "answer": out.get("answer") or out.get("result"),
        "sources": _fmt_sources(docs),
    }

def knowledge_rag_answer(
    question: str,
    session_id: str,
    k: int = 6,
    knowledge_paths: List[str] | None = None,
) -> Dict[str, Any]:
    """
    RAG over handbook/policy (no account data).
    """
    llm = _llm_from_cfg()
    kn_index = ensure_knowledge_index(knowledge_paths or _knowledge_paths_from_cfg())
    retriever = kn_index.as_langchain_retriever(similarity_top_k=k)

    chain = _crc(llm, retriever, session_id)
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    return {
        "answer": out.get("answer") or out.get("result"),
        "sources": _fmt_sources(docs),
    }