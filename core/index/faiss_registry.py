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
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "FAISS is required. Install with: pip install faiss-cpu"
    ) from e

# We use REST so we don't force the 'openai' Python SDK dependency.
import httpx


# =============================================================================
# Embedding client (OpenAI-compatible; easily swappable to Qwen later)
# =============================================================================

class Embedder:
    """
    Minimal HTTP client for embeddings.

    Defaults:
      provider = "openai"
      api_base = env OPENAI_API_BASE or "https://api.openai.com/v1"
      api_key  = env OPENAI_API_KEY
      model    = "text-embedding-3-large"
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

        # Strip and fail fast
        k = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not k:
            raise RuntimeError("Missing embedding API key. Set OPENAI_API_KEY or pass api_key=...")
        self.api_key = k

        base = (api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").strip()
        if base and not (base.startswith("http://") or base.startswith("https://")):
            base = "https://" + base
        self.api_base = base.rstrip("/")

        self.timeout = float(timeout)
        self.batch_size = int(batch_size)

    # ---- public --------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts and return an (N, D) float32 matrix.
        Automatically normalizes (L2) for cosine similarity with IndexFlatIP.
        """
        if not texts:
            return np.zeros((0, 1536), dtype="float32")

        # Batch to avoid request size limits
        vecs: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            vecs.extend(self._embed_batch(chunk))

        arr = np.array(vecs, dtype="float32")
        # L2 normalize -> cosine similarity with inner product
        faiss.normalize_L2(arr)
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        out = self.embed([text])
        return out[0] if out.shape[0] else np.zeros((self.dim_guess(),), dtype="float32")

    # ---- internals -----------------------------------------------------------

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            return self._embed_openai(texts)
        # You can add elif self.provider == "qwen" later, calling your Qwen endpoint.
        raise NotImplementedError(f"Unsupported provider: {self.provider}")

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.api_base}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        print("[EMBED DEBUG]", {"base": self.api_base, "model": self.model, "key_present": bool(self.api_key)})
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return [d["embedding"] for d in data["data"]]

    @staticmethod
    def dim_guess() -> int:
        # Common dims; this is only used if we need a placeholder vector.
        return 3072  # text-embedding-3-large is 3072 dims


# =============================================================================
# IO helpers
# =============================================================================

def _read_json_objects(path: Path) -> List[Dict[str, Any]]:
    """
    Accepts:
      - JSON array file
      - JSONL file (one object per line)
      - JSON object with top-level list under keys like 'data', 'items', 'transactions'
    Returns a list[dict].
    """
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        # Try JSONL
        objs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                objs.append(json.loads(line))
        return objs

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for k in ("data", "items", "transactions", "rows"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        # Single row dict
        return [data]

    return []


def _read_text_or_pdf(path: Path) -> str:
    """
    Reads .md/.txt as text; extracts PDF with PyPDF2 if installed.
    On failure, returns empty string.
    """
    if not path.exists():
        return ""

    if path.suffix.lower() == ".pdf":
        try:
            from PyPDF2 import PdfReader  # pip install pypdf2
        except Exception:
            return ""
        try:
            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    parts.append("")
            text = "\n".join(parts)
            return " ".join(text.split())
        except Exception:
            return ""

    # default text read
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _default_row_text(row: Dict[str, Any]) -> str:
    """
    Fallback text function that flattens a JSON object into a compact string.
    """
    pieces = []
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            try:
                v_str = json.dumps(v, ensure_ascii=False)
            except Exception:
                v_str = str(v)
        else:
            v_str = str(v)
        pieces.append(f"{k}={v_str}")
    return " | ".join(pieces)


def _chunk_text(t: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    t = " ".join((t or "").split())
    if not t:
        return []
    out = []
    i = 0
    step = max_chars - overlap
    if step <= 0:
        step = max_chars
    while i < len(t):
        out.append(t[i : i + max_chars])
        i += step
    return out


# =============================================================================
# On-disk index layout helpers
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
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    out.append({"_raw": line})
    return out


# =============================================================================
# Index builders
# =============================================================================

def _build_faiss(vectors: np.ndarray, metric: str = "ip") -> faiss.Index:
    """
    Build a FAISS index for given vectors.
    metric: "ip" (inner product; use with normalized vectors = cosine) or "l2"
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array (N, D)")

    n, d = vectors.shape
    if metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        index = faiss.IndexFlatIP(d)
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
    """
    Index a JSON array / JSONL of objects (rows) for semantic search.
    Stores:
      - {domain}.index       : FAISS index
      - {domain}_meta.json   : metadata
      - {domain}_rows.jsonl  : source rows (payloads)
      - {domain}_texts.jsonl : text used for embeddings (parallel order)
    """
    p = Path(path)
    rows = _read_json_objects(p)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    text_fn = text_fn or _default_row_text
    texts = [text_fn(r) for r in rows]

    emb = embedder or Embedder()
    vecs = emb.embed(texts)  # normalized

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
    """
    Index a text/markdown/PDF file by chunking into passage texts.
    Stores:
      - {domain}.index
      - {domain}_meta.json
      - {domain}_rows.jsonl  : {"chunk": i, "offset": start_char}
      - {domain}_texts.jsonl : {"text": "..."}
    """
    p = Path(path)
    text = _read_text_or_pdf(p)
    if not text.strip():
        raise ValueError(f"Empty or unreadable: {path}")

    chunks = _chunk_text(text, max_chars=max_chars, overlap=overlap)
    rows = [{"chunk": i, "offset": i * (max_chars - overlap)} for i in range(len(chunks))]
    texts = chunks

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
# Query helpers
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
    """
    Query a domain index with a natural-language string.
    Returns a list of {score, text, payload, idx}.
    Score is inner-product on normalized vectors => cosine similarity in [0, 1].
    """
    index, meta, rows, texts = _load_index(domain, index_dir)
    emb = embedder or Embedder()
    qv = emb.embed_one(query).reshape(1, -1)
    D, I = index.search(qv, min(top_k, len(texts)))
    scores = D[0]
    idxs = I[0]

    out: List[Dict[str, Any]] = []
    for s, i in zip(scores, idxs):
        if i < 0 or i >= len(texts):
            continue
        out.append({
            "score": float(s),
            "text": texts[i],
            "payload": rows[i] if i < len(rows) else {},
            "idx": int(i),
            "meta": meta,
        })
    return out


# =============================================================================
# Optional: small registry class used by startup.py (semantic + json)
# =============================================================================

class FaissRegistry:
    """
    Small convenience wrapper used by startup.py and anywhere else you want to
    manage multiple domain indexes with a consistent API.
    """

    def __init__(self, index_dir: str = "var/indexes") -> None:
        self.index_dir = index_dir

    # --- ensure / (re)build ---------------------------------------------------

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
        """
        Ensure an index exists. If 'rebuild=True', forces a rebuild.
        If is_text_chunks=True, expects rows_or_chunks like [{"text": "..."}].
        """
        ps = _paths(self.index_dir, domain)
        if (not rebuild) and ps["index"].exists() and ps["meta"].exists():
            # Already exists, return meta
            try:
                return json.loads(ps["meta"].read_text(encoding="utf-8"))
            except Exception:
                pass  # fall-through to rebuild

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

    # --- query ----------------------------------------------------------------

    def search(self, domain: str, query: str, k: int, embedder: Optional[Embedder] = None) -> List[Dict[str, Any]]:
        return query_index(domain=domain, query=query, top_k=k, index_dir=self.index_dir, embedder=embedder)