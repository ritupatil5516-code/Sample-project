# src/core/runtime.py
from __future__ import annotations
from typing import Optional
import httpx
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.api.contextApp.memory.memory_store import MEMORY as _MEMORY

class RuntimeConfig(BaseModel):
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    embed_model: str = "text-embedding-3-large"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key: str = ""
    memory_window_k: int = 10

class AppRuntime:
    def __init__(self) -> None:
        self.cfg: Optional[RuntimeConfig] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._chat: Optional[ChatOpenAI] = None
        self._emb: Optional[OpenAIEmbeddings] = None

    async def startup(self, cfg: RuntimeConfig, http_client: Optional[httpx.AsyncClient] = None):
        self.cfg = cfg
        self._http = http_client or httpx.AsyncClient(timeout=30)
        # one shared chat model
        self._chat = ChatOpenAI(
            model=cfg.llm_model,
            temperature=0,
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
        )
        # one shared embedding model
        self._emb = OpenAIEmbeddings(
            model=cfg.embed_model,
            base_url=cfg.embed_base_url,
            api_key=cfg.embed_api_key,
        )
        # configure memory window; memory objects are created lazily per session
        _MEMORY.configure(window_k=cfg.memory_window_k)

    async def shutdown(self):
        if self._http:
            await self._http.aclose()

    # --- getters used across the app ---
    def chat(self) -> ChatOpenAI:
        assert self._chat is not None, "Runtime not started"
        return self._chat

    def embeddings(self) -> OpenAIEmbeddings:
        assert self._emb is not None, "Runtime not started"
        return self._emb

    def memory(self, session_id: str):
        return _MEMORY.get(session_id)

# module-level singleton and DI helper
RUNTIME = AppRuntime()

def get_runtime() -> AppRuntime:
    return RUNTIME