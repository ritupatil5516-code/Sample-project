# rag_chain.py
from __future__ import annotations
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from llama_index.core import VectorStoreIndex

from pathlib import Path

# local loaders
from src.api.contextApp.index.index_builder import (
    load_account_index,
    load_knowledge_index,
)
from src.api.contextApp.settings import Settings  # your tiny config module


# ---- LlamaIndex retriever → LangChain wrapper ----
class _LlamaIndexToLangchainRetriever(BaseRetriever):
    def __init__(self, li_index: VectorStoreIndex, k: int = 6):
        super().__init__()
        self._retriever = li_index.as_retriever(similarity_top_k=k)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        nodes = self._retriever.retrieve(query)
        docs: List[Document] = []
        for n in nodes:
            meta = dict(n.metadata or {})
            docs.append(Document(page_content=n.get_text(), metadata=meta))
        return docs


# ---- LLM factory (uses your Settings) ----
def _llm_from_cfg():
    cfg = Settings.llm  # expects .model, .api_base, .api_key or env fallback
    return ChatOpenAI(
        model=cfg.get("model", "gpt-4o-mini"),
        temperature=0,
        base_url=cfg.get("api_base"),
        api_key=cfg.get("api_key"),
    )


# ---- ensure retrievers (load persisted FAISS via LlamaIndex) ----
def ensure_account_retriever(account_id: str, k: int = 6) -> BaseRetriever:
    root = Path(Settings.index_root) / "accounts" / account_id / "llama"
    index = load_account_index(root)
    return _LlamaIndexToLangchainRetriever(index, k=k)

def ensure_knowledge_retriever(k: int = 6) -> BaseRetriever:
    root = Path(Settings.index_root) / "knowledge" / "llama"
    index = load_knowledge_index(root)
    return _LlamaIndexToLangchainRetriever(index, k=k)


# ---- Conversation chain factory ----
def _crc(llm, retriever: BaseRetriever, session_id: str):
    mem = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
    mem.chat_memory.session_id = session_id
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=mem, return_source_documents=True
    )


# ---- Public APIs called by execute.py ----
def account_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    retr = ensure_account_retriever(account_id, k=k)
    chain = _crc(llm, retr, session_id)
    out = chain.invoke({"question": question})
    srcs = [{"source": d.metadata.get("domain"), "snippet": d.page_content[:180]} for d in out.get("source_documents", [])]
    return {"answer": out.get("answer") or out.get("result"), "sources": srcs}

def knowledge_rag_answer(question: str, session_id: str, k: int = 6) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    retr = ensure_knowledge_retriever(k=k)
    chain = _crc(llm, retr, session_id)
    out = chain.invoke({"question": question})
    srcs = [{"source": "knowledge", "snippet": d.page_content[:180]} for d in out.get("source_documents", [])]
    return {"answer": out.get("answer") or out.get("result"), "sources": srcs}

def unified_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> Dict[str, Any]:
    """
    Blend account + knowledge (simple alternation: ask account first; if low confidence or empty,
    ask knowledge; or you can implement EnsembleRetriever if you prefer).
    """
    # Option A: two-step ask (simple, predictable)
    a = account_rag_answer(question, session_id, account_id, k=k)
    if a.get("answer"):
        return a
    # fallback: knowledge only
    return knowledge_rag_answer(question, session_id, k=k)