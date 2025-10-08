# core/index/faiss_registry.py
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable, Tuple

import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    raise RuntimeError("FAISS is required. Install with: pip install faiss-cpu") from e

# --- Primary embedding backend: LlamaIndex (same tech stack as other embeddings) ---
#     We use the OpenAIEmbedding class which supports base_url/api_key via env.
#     If llama_index is missing, we fall back to a simple HTTP REST client.
_LLAMA_OK = False
try:
    from llama_index.embeddings.openai import OpenAIEmbedding  # llama-index>=0.10
    _LLAMA_OK = True
except Exception:
    _LLAMA_OK = False

import httpx  # fallback only


# =============================================================================
# Unified Embedder (LlamaIndex first; HTTP REST fallback)
# =============================================================================

class Embedder:
    """
    Unifies embeddings behind one interface.
    Prefers LlamaIndex's OpenAIEmbedding (same tech as other embeddings in the app).
    Falls back to HTTP REST (OpenAI-compatible) if llama_index is not installed.

    Config mapping (from config['embeddings']):
      provider:  "openai" (default) | "qwen" (OpenAI-compatible)
      openai_base_url / qwen_base_url
      openai_model    / qwen_model
      openai_api_key_env (default OPENAI_API_KEY)
      qwen_api_key_env   (default QWEN_API_KEY)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 30.0,
        batch_size: int = 64,
    ) -> None:
        self.provider = (provider or "openai").lower()
        self.model = model or "text-embedding-3-large"
        self.timeout = float(timeout)
        self.batch_size = int(batch_size)

        # Key + base
        if self.provider == "qwen":
            # OpenAI-compatible (DashScope compatible-mode)
            env_key = (api_key or os.getenv("QWEN_API_KEY") or "").strip()
            base = (api_base or os.getenv("QWEN_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        else:
            env_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
            base = (api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").strip()

        if not env_key:
            raise RuntimeError("Missing embedding API key. Set OPENAI_API_KEY (or QWEN_API_KEY) or pass api_key=...")

        if base and not (base.startswith("http://") or base.startswith("https://")):
            base = "https://" + base
        self.api_key = env_key
        self.api_base = base.rstrip("/")

        # Prepare LlamaIndex embedding backend if available
        self._li_embed: Optional[OpenAIEmbedding] = None
        if _LLAMA_OK:
            # LlamaIndex reads OpenAI settings from env variables:
            #  - OPENAI_API_KEY
            #  - OPENAI_API_BASE (if using a compatible endpoint)
            # Set the right envs for either provider.
            if self.provider == "qwen":
                os.environ["OPENAI_API_KEY"] = self.api_key
                os.environ["OPENAI_API_BASE"] = self.api_base
            else:
                os.environ["OPENAI_API_KEY"] = self.api_key
                os.environ["OPENAI_API_BASE"] = self.api_base

            try:
                self._li_embed = OpenAIEmbedding(model=self.model)
            except Exception:
                self._li_embed = None  # fallback to REST

    # ---- public --------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim_guess()), dtype="float32")

        # Prefer LlamaIndex backend
        if self._li_embed is not None:
            vecs = self._embed_llama_index(texts)
        else:
            vecs = []
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                vecs.extend(self._embed_rest(chunk))

        arr = np.array(vecs, dtype="float32")
        faiss.normalize_L2(arr)  # cosine via inner product
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        out = self.embed([text])
        return out[0] if out.shape[0] else np.zeros((self.dim_guess(),), dtype="float32")

    # ---- backends ------------------------------------------------------------

    def _embed_llama_index(self, texts: List[str]) -> List[List[float]]:
        # LlamaIndex returns Python lists of floats already
        return self._li_embed.get_text_embedding_batch(texts)  # type: ignore

    def _embed_rest(self, texts: List[str]) -> List[List[float]]:
        # OpenAI-compatible REST
        url = f"{self.api_base}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return [d["embedding"] for d in data["data"]]

    @staticmethod
    def dim_guess() -> int:
        # text-embedding-3-large = 3072; safe default
        return 3072


# =============================================================================
# Helper to build embedder from config
# =============================================================================

def build_embedder_from_config(cfg: Dict[str, Any]) -> Embedder:
    emb_cfg = cfg.get("embeddings", {}) or {}
    provider = (emb_cfg.get("provider") or "openai").strip().lower()

    if provider == "qwen":
        base = (emb_cfg.get("qwen_base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        model = (emb_cfg.get("qwen_model") or "qwen3-embedding").strip()
        key_env = (emb_cfg.get("qwen_api_key_env") or "QWEN_API_KEY").strip()
    else:
        base = (emb_cfg.get("openai_base_url") or "https://api.openai.com/v1").strip()
        model = (emb_cfg.get("openai_model") or "text-embedding-3-large").strip()
        key_env = (emb_cfg.get("openai_api_key_env") or "OPENAI_API_KEY").strip()

    api_key = (os.getenv(key_env) or "").strip()
    if not api_key:
        raise RuntimeError(f"[faiss_registry] Missing embedding API key. Set env var {key_env}")

    return Embedder(provider=provider, model=model, api_key=api_key, api_base=base)


# =============================================================================
# IO helpers (unchanged)
# =============================================================================

def _read_json_objects(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        # JSONL fallback
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                out.append(json.loads(s))
        return out

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "items", "transactions", "rows"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        return [data]
    return []


def _read_text_or_pdf(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except Exception:
            return ""
        try:
            pages = []
            reader = PdfReader(str(path))
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return " ".join("\n".join(pages).split())
        except Exception:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _default_row_text(row: Dict[str, Any]) -> str:
    parts = []
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            try:
                v_str = json.dumps(v, ensure_ascii=False)
            except Exception:
                v_str = str(v)
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return " | ".join(parts)


def _chunk_text(t: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    t = " ".join((t or "").split())
    if not t:
        return []
    out, i = [], 0
    step = max_chars - overlap if (max_chars - overlap) > 0 else max_chars
    while i < len(t):
        out.append(t[i : i + max_chars])
        i += step
    return out


# =============================================================================
# On-disk index layout helpers (unchanged)
# =============================================================================

def _paths(index_dir: str, domain: str) -> Dict[str, Path]:
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "index": d / f"{domain}.index",
        "meta": d / f"{domain}_meta.json",
        "rows": d / f"{domain}_rows.jsonl",
        "texts": d / f"{domain}_texts.jsonl",
    }


def _write_rows_jsonl(path: Path, objs: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            n += 1
    return n


def _read_rows_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    out.append(json.loads(s))
                except Exception:
                    out.append({"_raw": s})
    return out


# =============================================================================
# Index builders (unchanged except they use the new Embedder)
# =============================================================================

def _build_faiss(vectors: np.ndarray, metric: str = "ip") -> faiss.Index:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array (N, D)")
    _, d = vectors.shape
    index = faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def index_json_file(
    path: str,
    domain: str,
    embedder: Optional[Embedder] = None,
    index_dir: str = "var/indexes",
    text_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    metric: str = "ip",
) -> Dict[str, Any]:
    p = Path(path)
    rows = _read_json_objects(p)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    text_fn = text_fn or _default_row_text
    texts = [text_fn(r) for r in rows]

    emb = embedder or Embedder()
    vecs = emb.embed(texts)
    index = _build_faiss(vecs, metric=metric)

    ps = _paths(index_dir, domain)
    faiss.write_index(index, str(ps["index"]))
    _write_rows_jsonl(ps["rows"], rows)
    _write_rows_jsonl(ps["texts"], [{"text": t} for t in texts])

    meta = {
        "domain": domain,
        "source": str(p),
        "count": len(rows),
        "dim": int(vecs.shape[1]),
        "metric": metric,
        "provider": emb.provider,
        "model": emb.model,
        "created_at": int(time.time()),
        "type": "json",
    }
    ps["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def index_text_file(
    path: str,
    domain: str,
    embedder: Optional[Embedder] = None,
    index_dir: str = "var/indexes",
    metric: str = "ip",
    max_chars: int = 1500,
    overlap: int = 150,
) -> Dict[str, Any]:
    p = Path(path)
    text = _read_text_or_pdf(p)
    if not text.strip():
        raise ValueError(f"Empty or unreadable: {path}")

    chunks = _chunk_text(text, max_chars=max_chars, overlap=overlap)
    rows = [{"chunk": i, "offset": i * (max_chars - overlap)} for i in range(len(chunks))]
    emb = embedder or Embedder()
    vecs = emb.embed(chunks)
    index = _build_faiss(vecs, metric=metric)

    ps = _paths(index_dir, domain)
    faiss.write_index(index, str(ps["index"]))
    _write_rows_jsonl(ps["rows"], rows)
    _write_rows_jsonl(ps["texts"], [{"text": t} for t in chunks])

    meta = {
        "domain": domain,
        "source": str(p),
        "count": len(chunks),
        "dim": int(vecs.shape[1]),
        "metric": metric,
        "provider": emb.provider,
        "model": emb.model,
        "created_at": int(time.time()),
        "type": "text",
        "chunking": {"max_chars": max_chars, "overlap": overlap},
    }
    ps["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# =============================================================================
# Query helpers (unchanged)
# =============================================================================

def _load_index(domain: str, index_dir: str = "var/indexes") -> Tuple[faiss.Index, Dict[str, Any], List[Dict[str, Any]], List[str]]:
    ps = _paths(index_dir, domain)
    if not ps["index"].exists():
        raise FileNotFoundError(f"No FAISS index found for domain '{domain}' at {ps['index']}")
    index = faiss.read_index(str(ps["index"]))

    meta = {}
    if ps["meta"].exists():
        try:
            meta = json.loads(ps["meta"].read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    rows = _read_rows_jsonl(ps["rows"])
    texts = [d.get("text", "") for d in _read_rows_jsonl(ps["texts"])]
    return index, meta, rows, texts


def query_index(
    domain: str,
    query: str,
    top_k: int = 3,
    index_dir: str = "var/indexes",
    embedder: Optional[Embedder] = None,
) -> List[Dict[str, Any]]:
    index, meta, rows, texts = _load_index(domain, index_dir)
    emb = embedder or Embedder()
    qv = emb.embed_one(query).reshape(1, -1)
    k = min(max(1, top_k), len(texts))
    D, I = index.search(qv, k)
    out: List[Dict[str, Any]] = []
    for s, i in zip(D[0], I[0]):
        if 0 <= i < len(texts):
            out.append({
                "score": float(s),
                "text": texts[i],
                "payload": rows[i] if i < len(rows) else {},
                "idx": int(i),
                "meta": meta,
            })
    return out


# =============================================================================
# Convenience registry
# =============================================================================

class FaissRegistry:
    def __init__(self, index_dir: str = "var/indexes") -> None:
        self.index_dir = index_dir

    def ensure(
        self,
        domain: str,
        rows_or_chunks: List[Dict[str, Any]],
        embedder: Embedder,
        text_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        metric: str = "ip",
        rebuild: bool = False,
        is_text_chunks: bool = False,
    ) -> Dict[str, Any]:
        ps = _paths(self.index_dir, domain)
        if (not rebuild) and ps["index"].exists() and ps["meta"].exists():
            try:
                return json.loads(ps["meta"].read_text(encoding="utf-8"))
            except Exception:
                pass

        if is_text_chunks:
            texts = [d.get("text", "") for d in rows_or_chunks]
            rows = [{"chunk": i} for i in range(len(texts))]
        else:
            text_fn = text_fn or _default_row_text
            texts = [text_fn(r) for r in rows_or_chunks]
            rows = rows_or_chunks

        vecs = embedder.embed(texts)
        index = _build_faiss(vecs, metric=metric)
        faiss.write_index(index, str(ps["index"]))
        _write_rows_jsonl(ps["rows"], rows)
        _write_rows_jsonl(ps["texts"], [{"text": t} for t in texts])

        meta = {
            "domain": domain,
            "count": len(texts),
            "dim": int(vecs.shape[1]),
            "metric": metric,
            "provider": embedder.provider,
            "model": embedder.model,
            "created_at": int(time.time()),
            "type": "text" if is_text_chunks else "json",
        }
        ps["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def search(self, domain: str, query: str, k: int, embedder: Optional[Embedder] = None) -> List[Dict[str, Any]]:
        return query_index(domain=domain, query=query, top_k=k, index_dir=self.index_dir, embedder=embedder)