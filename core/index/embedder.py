from typing import List, Optional
import os, httpx
def embed_texts(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    if not texts: return []
    base = (os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")
    key  = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = model or (os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small")
    if not key: raise RuntimeError("Missing OPENAI_API_KEY for embeddings")
    url = f"{base}/embeddings"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload, headers=headers); r.raise_for_status(); data = r.json()
    return [item["embedding"] for item in data["data"]]
