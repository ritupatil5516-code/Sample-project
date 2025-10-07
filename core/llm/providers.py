from __future__ import annotations
from typing import List, Dict
import os, httpx

class ChatLLM:
    def complete(self, messages: List[Dict[str,str]], model: str, temperature: float = 0) -> str:
        raise NotImplementedError

class OpenAIChat(ChatLLM):
    def __init__(self, base_url: str, api_key_env: str = "OPENAI_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env, "")

    def complete(self, messages: List[Dict[str,str]], model: str, temperature: float = 0) -> str:
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": temperature}
        with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
            r = client.post("/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

class LlamaChat(ChatLLM):
    def __init__(self, base_url: str, api_key_env: str = "LLAMA_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env, "")

    def complete(self, messages: List[Dict[str,str]], model: str, temperature: float = 0) -> str:
        if not self.api_key:
            raise RuntimeError("Missing LLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": temperature}
        with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
            r = client.post("/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

def build_llm_from_config(cfg: dict) -> ChatLLM:
    prov = (cfg.get("llm") or {}).get("provider","openai").lower()
    if prov == "openai":
        return OpenAIChat(base_url=(cfg["llm"]["base_url"]))
    elif prov == "llama":
        return LlamaChat(base_url=(cfg["llm"]["base_url"]))
    else:
        raise ValueError(f"Unknown LLM provider: {prov}")

def get_embedding_model():
    """
    Returns a callable that can embed a text using the configured model.
    """
    client = OpenAIChat(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("EMBED_MODEL", "text-embedding-3-large")

    def _embed(text: str):
        res = client.embeddings.create(model=model, input=text)
        return res.data[0].embedding

    return _embed