# rag_chain.py
from __future__ import annotations
from typing import Any, Dict, List, Iterable

# --- LangChain LLM + memory + CRChain ---
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document as LCDocument
from langchain.retrievers import BaseRetriever

# --- Your ingestion utilities (already working) ---
from core.retrieval.json_ingest import ensure_account_retriever
from core.retrieval.knowledge_ingest import ensure_knowledge_retriever


# -------------------------- LLM factory --------------------------

def _llm_from_cfg():
    # Keep whatever you had; this mirrors your screenshots
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ----------------- LlamaIndex -> LangChain adapter ----------------

class _LlamaIndexToLCRetriever(BaseRetriever):
    """Wrap a LlamaIndex BaseRetriever so LangChain CRChain can use it."""
    def __init__(self, li_retriever):
        super().__init__()
        self._r = li_retriever

    def _nodes_to_lc(self, nodes: Iterable[Any]) -> List[LCDocument]:
        docs: List[LCDocument] = []
        for obj in nodes:
            # LlamaIndex may return NodeWithScore or a Node; normalize
            node = getattr(obj, "node", obj)
            # text/content
            text = getattr(node, "text", None)
            if not text and hasattr(node, "get_content"):
                try:
                    text = node.get_content()
                except Exception:
                    text = ""
            text = text or ""
            # metadata
            meta = {}
            m = getattr(node, "metadata", None)
            if isinstance(m, dict):
                meta = dict(m)
            docs.append(LCDocument(page_content=text, metadata=meta))
        return docs

    # LangChain sync hook
    def get_relevant_documents(self, query: str) -> List[LCDocument]:  # type: ignore[override]
        try:
            nodes = self._r.retrieve(query)
        except TypeError:
            # Some retrievers use .query(...)
            nodes = self._r.query(query)
        return self._nodes_to_lc(nodes)

    # LangChain async hook (optional)
    async def aget_relevant_documents(self, query: str) -> List[LCDocument]:  # type: ignore[override]
        try:
            nodes = await self._r.aretrieve(query)  # if available
        except Exception:
            nodes = self._r.retrieve(query)
        return self._nodes_to_lc(nodes)


# ---------------------- memory helper ----------------------------

def _memory(session_id: str, k: int = 10):
    # Warning about deprecation is harmless; this class still works.
    # You can migrate later to the new LC memory API if you want.
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=k,
        chat_memory_key=session_id,  # keeps sessions separate
    )


# ---------------------- chain builder ----------------------------

def _cr_chain(llm, retriever: BaseRetriever, session_id: str):
    mem = _memory(session_id, k=10)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,               # <- must be a LangChain BaseRetriever
        memory=mem,
        return_source_documents=True,
    )


# ----------------- Public RAG entry points -----------------------

def unified_rag_answer(
    question: str,
    session_id: str,
    account_id: str,
    k: int = 6,
    **_ignore,
) -> Dict[str, Any]:
    """
    Hybrid retrieval: account JSONs (LlamaIndex) + knowledge (LlamaIndex),
    then adapt to LangChain CRChain with an adapter.
    """
    llm = _llm_from_cfg()

    # LlamaIndex retrievers (you already build these in your ingest)
    r_acc = ensure_account_retriever(account_id=account_id, k=k)
    r_kn  = ensure_knowledge_retriever(k=k)

    # Simple weighted blend: get top-k from each and concatenate.
    # CRChain needs a single retriever; we implement a tiny composite.
    class _CompositeLCRetriever(BaseRetriever):
        def __init__(self, r1, r2):
            super().__init__()
            self.r1 = _LlamaIndexToLCRetriever(r1)
            self.r2 = _LlamaIndexToLCRetriever(r2)
        def get_relevant_documents(self, query: str) -> List[LCDocument]:  # type: ignore[override]
            a = self.r1.get_relevant_documents(query)
            b = self.r2.get_relevant_documents(query)
            # naive blend; keep it simple
            take = max(1, min(len(a) + len(b), k))
            return (a + b)[:take]
        async def aget_relevant_documents(self, query: str) -> List[LCDocument]:  # type: ignore[override]
            a = await self.r1.aget_relevant_documents(query)
            b = await self.r2.aget_relevant_documents(query)
            take = max(1, min(len(a) + len(b), k))
            return (a + b)[:take]

    lc_retriever = _CompositeLCRetriever(r_acc, r_kn)
    chain = _cr_chain(llm, lc_retriever, session_id=session_id)

    out = chain.invoke({"question": question})
    src = out.get("source_documents") or []
    sources = [{"source": d.metadata.get("source") or d.metadata.get("file_path") or "",
                "snippet": (d.page_content or "")[:180]} for d in src[:4]]
    return {"answer": out.get("answer") or out.get("result"), "sources": sources}


def account_rag_answer(
    question: str,
    session_id: str,
    account_id: str,
    k: int = 6,
) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    r_acc = ensure_account_retriever(account_id=account_id, k=k)  # LlamaIndex retriever
    lc_ret = _LlamaIndexToLCRetriever(r_acc)                      # adapt
    chain = _cr_chain(llm, lc_ret, session_id=session_id)

    out = chain.invoke({"question": question})
    src = out.get("source_documents") or []
    sources = [{"source": d.metadata.get("source") or d.metadata.get("file_path") or "",
                "snippet": (d.page_content or "")[:180]} for d in src[:4]]
    return {"answer": out.get("answer") or out.get("result"), "sources": sources}


def knowledge_rag_answer(
    question: str,
    session_id: str,
    k: int = 6,
) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    r_kn = ensure_knowledge_retriever(k=k)   # LlamaIndex retriever
    lc_ret = _LlamaIndexToLCRetriever(r_kn)  # adapt
    chain = _cr_chain(llm, lc_ret, session_id=session_id)

    out = chain.invoke({"question": question})
    src = out.get("source_documents") or []
    sources = [{"source": d.metadata.get("source") or d.metadata.get("file_path") or "",
                "snippet": (d.page_content or "")[:180]} for d in src[:4]]
    return {"answer": out.get("answer") or out.get("result"), "sources": sources}