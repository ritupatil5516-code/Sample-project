# core/retrieval/rag_chain.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

# LangChain (conversation + ensemble)
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever

# LlamaIndex (load persisted FAISS indexes)
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore

# Your process-wide singletons (LLM, memory, cfg)
from src.core.runtime import RUNTIME


# --------------------------- LlamaIndex helpers ---------------------------

def _li_index_from_dir(persist_dir: str):
    """Load a LlamaIndex VectorStoreIndex from a FAISS persist_dir. Returns None if missing."""
    p = Path(persist_dir)
    if not p.exists():
        return None
    try:
        vs = FaissVectorStore.from_persist_dir(str(p))
        sc = StorageContext.from_defaults(vector_store=vs, persist_dir=str(p))
        return load_index_from_storage(sc)
    except Exception as e:
        print(f"[RAG] load_index_from_storage failed @ {persist_dir}: {e}")
        return None


def _lc_retriever_from_dir(persist_dir: str, k: int = 6):
    """Return a LangChain retriever adapted from a LlamaIndex retriever, or None."""
    index = _li_index_from_dir(persist_dir)
    if index is None:
        return None
    li_ret = index.as_retriever(similarity_top_k=int(k))
    try:
        return li_ret.as_langchain()  # <-- critical adapter for LC chains
    except Exception as e:
        print(f"[RAG] as_langchain() failed @ {persist_dir}: {e}")
        return None


# --------------------------- Public entry points ---------------------------

def unified_rag_answer(
    question: str,
    session_id: str,
    account_id: Optional[str],
    k: int = 6,
) -> Dict[str, Any]:
    """
    Conversational RAG using:
      - account index: var/indexes/accounts/{account_id}
      - knowledge index: var/indexes/knowledge
    LLM + memory come from RUNTIME. LlamaIndex retrievers are adapted to LC.
    """
    cfg = RUNTIME.cfg or {}
    chat = RUNTIME.chat()                 # shared ChatOpenAI (or OpenAI-compatible) from runtime
    memory = RUNTIME.memory(session_id)   # ConversationBufferWindowMemory from runtime

    idx_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    acc_dir = (Path(idx_dir) / "accounts" / str(account_id)) if account_id else None
    kn_dir  = Path(idx_dir) / "knowledge"

    retrievers: List = []
    # account retriever (if available)
    if acc_dir:
        r_acc = _lc_retriever_from_dir(acc_dir.as_posix(), k=k)
        if r_acc:
            retrievers.append(("acc", r_acc))
        else:
            print(f"[RAG] missing account retriever @ {acc_dir}")

    # knowledge retriever (handbook + agreement)
    r_kn = _lc_retriever_from_dir(kn_dir.as_posix(), k=k)
    if r_kn:
        retrievers.append(("kn", r_kn))
    else:
        print(f"[RAG] missing knowledge retriever @ {kn_dir}")

    if not retrievers:
        return {
            "answer": "I don’t have any indexed data available yet to answer this.",
            "sources": [],
            "error": "no_retriever",
        }

    # combine (if both present) or use the single one
    if len(retrievers) == 1:
        retriever = retrievers[0][1]
    else:
        # prefer account evidence a bit more than generic knowledge
        retriever = EnsembleRetriever(
            retrievers=[r for _, r in retrievers],
            weights=[0.7, 0.3] if len(retrievers) == 2 else None,
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    sources = []
    for d in docs[:6]:
        try:
            md = getattr(d, "metadata", {}) or {}
            sources.append({
                "source": md.get("source") or md.get("file_path") or md.get("path") or "doc",
                "snippet": (d.page_content or "")[:220],
            })
        except Exception:
            pass

    answer = out.get("answer") or out.get("result") or ""
    return {"answer": answer, "sources": sources}


def account_rag_answer(
    question: str,
    session_id: str,
    account_id: str,
    k: int = 6,
) -> Dict[str, Any]:
    """RAG limited to a single account index."""
    cfg = RUNTIME.cfg or {}
    chat = RUNTIME.chat()
    memory = RUNTIME.memory(session_id)
    idx_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    acc_dir = Path(idx_dir) / "accounts" / str(account_id)

    r_acc = _lc_retriever_from_dir(acc_dir.as_posix(), k=k)
    if not r_acc:
        return {"answer": "I don’t see an index for this account yet.", "sources": [], "error": "no_account_index"}

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=r_acc, memory=memory, return_source_documents=True
    )
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    sources = []
    for d in docs[:6]:
        try:
            md = getattr(d, "metadata", {}) or {}
            sources.append({
                "source": md.get("source") or md.get("file_path") or md.get("path") or "doc",
                "snippet": (d.page_content or "")[:220],
            })
        except Exception:
            pass

    return {"answer": out.get("answer") or out.get("result") or "", "sources": sources}


def knowledge_rag_answer(
    question: str,
    session_id: str,
    k: int = 6,
) -> Dict[str, Any]:
    """RAG limited to the knowledge index (handbook + policy/agreement)."""
    cfg = RUNTIME.cfg or {}
    chat = RUNTIME.chat()
    memory = RUNTIME.memory(session_id)
    idx_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    kn_dir = Path(idx_dir) / "knowledge"

    r_kn = _lc_retriever_from_dir(kn_dir.as_posix(), k=k)
    if not r_kn:
        return {"answer": "Knowledge index not found yet.", "sources": [], "error": "no_knowledge_index"}

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=r_kn, memory=memory, return_source_documents=True
    )
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    sources = []
    for d in docs[:6]:
        try:
            md = getattr(d, "metadata", {}) or {}
            sources.append({
                "source": md.get("source") or md.get("file_path") or md.get("path") or "doc",
                "snippet": (d.page_content or "")[:220],
            })
        except Exception:
            pass

    return {"answer": out.get("answer") or out.get("result") or "", "sources": sources}