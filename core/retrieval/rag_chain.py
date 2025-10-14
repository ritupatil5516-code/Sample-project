# core/retrieval/rag_chain.py
from pathlib import Path
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from core.retrieval.json_ingest import load_account_index
from core.retrieval.knowledge_ingest import load_knowledge_index
from core.memory.memory_store import MEMORY

INDEX_STORE_DIR = Path("src/api/contextApp/indexesstore")

def _llm_from_cfg():
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def _fmt_sources(docs: List[Any]) -> List[Dict[str, str]]:
    out = []
    for d in (docs or [])[:5]:
        meta = getattr(d, "metadata", {}) or {}
        out.append({
            "source": meta.get("source") or meta.get("file_path") or "doc",
            "snippet": (getattr(d, "page_content", "") or "")[:180],
        })
    return out

def ensure_account_retriever(account_id: str, k: int = 6):
    acc_li_dir = INDEX_STORE_DIR / "accounts" / account_id / "llama"
    idx = load_account_index(str(acc_li_dir))
    return idx.as_retriever(similarity_top_k=k)

def ensure_knowledge_retriever(k: int = 6):
    kn_li_dir = INDEX_STORE_DIR / "knowledge" / "llama"
    idx = load_knowledge_index(str(kn_li_dir))
    return idx.as_retriever(similarity_top_k=k)

def _crc(llm, retriever, session_id: str):
    mem = MEMORY.get(session_id)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=mem, return_source_documents=True
    )

def account_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    retr = ensure_account_retriever(account_id, k=k)
    chain = _crc(llm, retr, session_id)
    out = chain.invoke({"question": question})
    return {"answer": out.get("answer") or out.get("result"), "sources": _fmt_sources(out.get("source_documents"))}

def knowledge_rag_answer(question: str, session_id: str, k: int = 6) -> Dict[str, Any]:
    llm = _llm_from_cfg()
    retr = ensure_knowledge_retriever(k=k)
    chain = _crc(llm, retr, session_id)
    out = chain.invoke({"question": question})
    return {"answer": out.get("answer") or out.get("result"), "sources": _fmt_sources(out.get("source_documents"))}

def unified_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> Dict[str, Any]:
    # simple fallback: try account first then knowledge
    acc = account_rag_answer(question, session_id, account_id, k=k)
    if acc.get("answer"):
        return acc
    return knowledge_rag_answer(question, session_id, k=k)