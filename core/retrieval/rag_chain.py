# core/retrieval/rag_chain.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI

from llama_index.core import load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext

from core.memory.memory_store import MEMORY  # your existing memory
from core.retrieval.index_builder import INDEXES_DIR, KNOWLEDGE_INDEX

def _llm_from_cfg():
    # very light wrapper – uses your OPENAI_API_KEY and model from env or config
    # keep it simple; you already have a providers.py if you want to route here
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)

def _load_index(persist_dir: Path):
    if not persist_dir.exists():
        raise FileNotFoundError(f"Missing index at {persist_dir}. Build indexes at startup.")
    sc = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(sc)

def ensure_account_retriever(account_id: str, top_k: int = 6):
    idx = _load_index(INDEXES_DIR / account_id)
    return idx.as_langchain_retriever(similarity_top_k=top_k)

def ensure_knowledge_retriever(top_k: int = 6):
    idx = _load_index(KNOWLEDGE_INDEX)
    return idx.as_langchain_retriever(similarity_top_k=top_k)

def _crc(llm, retriever, session_id: str):
    mem = MEMORY.get(session_id)  # returns a ConversationBufferWindowMemory(k=10, …)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=mem,
        return_source_documents=True
    )

def _fmt_sources(docs) -> List[Dict[str, str]]:
    out = []
    for d in (docs or [])[:5]:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("file_path") or meta.get("path") or meta.get("domain") or "source"
        txt = getattr(d, "page_content", "") or ""
        out.append({"source": str(src), "snippet": txt[:180]})
    return out

# ---------- public APIs used by execute.py ----------

def unified_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    r_acc = ensure_account_retriever(account_id, top_k=k)
    r_kn  = ensure_knowledge_retriever(top_k=k)
    # weight account higher than knowledge
    retriever = EnsembleRetriever(retrievers=[r_acc, r_kn], weights=[0.7, 0.3])
    chain = _crc(llm, retriever, session_id=session_id)
    out = chain.invoke({"question": question})
    return {"answer": out.get("answer") or out.get("result"), "sources": _fmt_sources(out.get("source_documents"))}

def account_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    retriever = ensure_account_retriever(account_id, top_k=k)
    chain = _crc(llm, retriever, session_id=session_id)
    out = chain.invoke({"question": question})
    return {"answer": out.get("answer") or out.get("result"), "sources": _fmt_sources(out.get("source_documents"))}

def knowledge_rag_answer(question: str, session_id: str, k: int = 6) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    retriever = ensure_knowledge_retriever(top_k=k)
    chain = _crc(llm, retriever, session_id=session_id)
    out = chain.invoke({"question": question})
    return {"answer": out.get("answer") or out.get("result"), "sources": _fmt_sources(out.get("source_documents"))}