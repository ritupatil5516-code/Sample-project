# core/retrieval/rag_chain.py
from __future__ import annotations
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from core.memory.memory_store import MEMORY
from core.retrieval.json_ingest import ensure_account_retriever
from core.retrieval.knowledge_ingest import ensure_knowledge_retriever

def _llm_from_cfg():
    import os, yaml
    cfg = yaml.safe_load(open("config/app.yaml","r",encoding="utf-8").read()) if True else {}
    llm_cfg = (cfg.get("llm") or {})
    model = (llm_cfg.get("model") or "gpt-4o-mini")
    base  = (llm_cfg.get("api_base") or "https://api.openai.com/v1")
    key   = (llm_cfg.get("api_key") or
             (llm_cfg.get("api_key_env") and os.getenv(llm_cfg.get("api_key_env"),"")) or "")
    return ChatOpenAI(model_name=model, temperature=0, base_url=base, api_key=key)

def _crc(llm, retriever, session_id:str):
    mem = MEMORY.get(session_id)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=mem, return_source_documents=True
    )

def account_rag_answer(question:str, session_id:str, account_id:str, k:int=6) -> Dict[str,Any]:
    llm = _llm_from_cfg()
    retr = ensure_account_retriever(account_id, k=k)
    chain = _crc(llm, retr, session_id)
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    sources = [{"source": d.metadata.get("source"), "snippet": d.page_content[:180]} for d in docs[:5]]
    return {"answer": out.get("answer") or out.get("result"), "sources": sources}

def knowledge_rag_answer(question:str, session_id:str, k:int=6) -> Dict[str,Any]:
    llm = _llm_from_cfg()
    retr = ensure_knowledge_retriever(k=k)
    chain = _crc(llm, retr, session_id)
    out = chain.invoke({"question": question})
    docs = out.get("source_documents") or []
    sources = [{"source": d.metadata.get("source"), "snippet": d.page_content[:180]} for d in docs[:5]]
    return {"answer": out.get("answer") or out.get("result"), "sources": sources}