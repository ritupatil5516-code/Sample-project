# src/api/contextApp/runtime.py
from __future__ import annotations
from typing import Dict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory

class _Runtime:
    llm: Optional[ChatOpenAI] = None
    embeddings: Optional[OpenAIEmbeddings] = None
    memories: Dict[str, ConversationBufferWindowMemory] = {}
    history_k: int = 10

RUNTIME = _Runtime()

def init_llm(*, model: str, api_key: str, api_base: Optional[str] = None, temperature: float = 0.0):
    RUNTIME.llm = ChatOpenAI(model=model, api_key=api_key, base_url=api_base, temperature=temperature)

def init_embeddings(*, model: str, api_key: str, api_base: Optional[str] = None):
    RUNTIME.embeddings = OpenAIEmbeddings(model=model, api_key=api_key, base_url=api_base)

def set_memory_window(k: int = 10):
    RUNTIME.history_k = max(1, int(k))

def get_llm() -> ChatOpenAI:
    assert RUNTIME.llm, "LLM not initialized"
    return RUNTIME.llm

def get_embeddings() -> OpenAIEmbeddings:
    assert RUNTIME.embeddings, "Embeddings not initialized"
    return RUNTIME.embeddings

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    mem = RUNTIME.memories.get(session_id)
    if not mem:
        mem = ConversationBufferWindowMemory(k=RUNTIME.history_k, return_messages=True, memory_key="chat_history")
        RUNTIME.memories[session_id] = mem
    return mem