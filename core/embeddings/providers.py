from __future__ import annotations
from typing import List
import os, httpx, json

class TextEmbedder:
    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        raise NotImplementedError

class OpenAIEmbeddings(TextEmbedder):
    def __init__(self, base_url: str, api_key_env: str = "OPENAI_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env, "")

    def embed(self, texts: List[str], model: str) -> List[List[float]]:

        if not texts: return []
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for embeddings")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "input": texts}
        with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
            r = client.post("/embeddings", headers=headers, json=payload)
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"OpenAI embeddings failed: {r.status_code} {r.text}") from e
            data = r.json()
            if "data" not in data:
                raise RuntimeError(f"Unexpected embeddings response: {json.dumps(data)[:300]}")
            return [row["embedding"] for row in data["data"]]

class QwenEmbeddings(TextEmbedder):
    def __init__(self, base_url: str, api_key_env: str = "QWEN_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env, "")

    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        if not texts: return []
        if not self.api_key:
            raise RuntimeError("Missing QWEN_API_KEY for embeddings")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "input": texts}
        with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
            r = client.post("/embeddings", headers=headers, json=payload)
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Qwen embeddings failed: {r.status_code} {r.text}") from e
            data = r.json()
            if "data" not in data:
                raise RuntimeError(f"Unexpected embeddings response: {json.dumps(data)[:300]}")
            return [row["embedding"] for row in data["data"]]

def build_embedder_from_config(cfg: dict):
    e = cfg.get("embeddings") or {}
    prov = (e.get("provider") or "openai").lower()
    if prov == "openai":
        return OpenAIEmbeddings(base_url=e.get("openai_base_url","https://api.openai.com/v1"), api_key_env=e.get("openai_api_key_env","OPENAI_API_KEY"))
    elif prov == "qwen":
        return QwenEmbeddings(base_url=e.get("qwen_base_url","https://dashscope.aliyuncs.com/compatible-mode/v1"), api_key_env=e.get("qwen_api_key_env","QWEN_API_KEY"))
    else:
        raise ValueError(f"Unknown embeddings provider: {prov}")
