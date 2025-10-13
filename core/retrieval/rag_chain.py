# core/retrieval/rag_chain.py
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from core.providers import read_cfg, build_langchain_llm
from core.memory.memory_store import MEMORY
from core.retrieval.llama_indices import (
    ensure_account_index, ensure_knowledge_index,
    account_retriever, knowledge_retriever
)

def unified_rag_answer(question: str, session_id: str, account_id: str,
                       txns, pays, stmts, acct,
                       knowledge_paths: list[str],
                       top_k: int = 5) -> dict:
    cfg = read_cfg()
    llm = build_langchain_llm(cfg, temperature=0.0)
    mem = MEMORY.get(session_id)

    acc_idx = ensure_account_index(account_id, txns, pays, stmts, acct)
    kn_idx  = ensure_knowledge_index(knowledge_paths)

    # combine retrievers: ask account first, fall back to knowledge
    from langchain.retrievers import EnsembleRetriever
    r1 = acc_idx.as_langchain_retriever(similarity_top_k=top_k)
    r2 = kn_idx.as_langchain_retriever(similarity_top_k=top_k)
    retriever = EnsembleRetriever(retrievers=[r1, r2], weights=[0.7, 0.3])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=mem, return_source_documents=True
    )
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    sources = [{"source": d.metadata.get("source") or d.metadata.get("file_path") or "",
                "snippet": (getattr(d, "page_content", "") or "")[:180]} for d in docs[:5]]
    return {"answer": out.get("answer") or out.get("result"), "sources": sources}