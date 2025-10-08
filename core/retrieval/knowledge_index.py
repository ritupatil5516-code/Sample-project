# core/retrieval/knowledge_faiss.py
"""
Unified knowledge index (handbook + any .md in data/knowledge) using LlamaIndex + FAISS.

Exports:
- ensure_knowledge_index(knowledge_dir="data/knowledge", ...)
- query_knowledge(query, ...)

This complements the policy_index.py (which focuses on the agreement PDF).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import yaml

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding


# ------------------------------ config helpers --------------------------------

def _cfg(config_path: str = "config/app.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        return {}
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _build_embed_model(cfg: Dict[str, Any]):
    emb = (cfg.get("embeddings") or {})
    provider = (emb.get("provider") or "openai").strip().lower()

    if provider == "qwen":
        # Wire up Qwen embedding here if/when you add a LlamaIndex-compatible class.
        raise NotImplementedError("Qwen embeddings via LlamaIndex not yet wired.")
    else:
        base = (emb.get("openai_base_url") or "https://api.openai.com/v1").strip()
        model = (emb.get("openai_model") or "text-embedding-3-large").strip()
        key_env = (emb.get("openai_api_key_env") or "OPENAI_API_KEY").strip()
        api_key = (os.getenv(key_env) or "").strip()
        if not api_key:
            raise RuntimeError(f"[knowledge_faiss] Missing embedding API key. Set env var {key_env}.")
        Settings.embed_model = OpenAIEmbedding(model=model, api_base=base)


def _probe_dim() -> int:
    vec = Settings.embed_model.get_text_embedding("dim_probe")
    return len(vec)


def _paths(persist_dir: str) -> Dict[str, Path]:
    d = Path(persist_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "faiss": d / "faiss.index",
        "docstore": d / "docstore.json",
        "meta": d / "knowledge_meta.json",
    }


# ------------------------------ build / ensure --------------------------------

def ensure_knowledge_index(
    knowledge_dir: str = "data/knowledge",
    *,
    config_path: str = "config/app.yaml",
    persist_dir: Optional[str] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Index all markdown/text files in data/knowledge (e.g., handbook.md).
    Persists under var/indexes/knowledge/.
    """
    cfg = _cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    persist_dir = persist_dir or str(Path(idx_cfg.get("dir", "var/indexes")) / "knowledge")

    _build_embed_model(cfg)
    dim = _probe_dim()

    ps = _paths(persist_dir)
    if (not rebuild) and ps["faiss"].exists() and ps["docstore"].exists():
        try:
            return json.loads(ps["meta"].read_text(encoding="utf-8"))
        except Exception:
            pass

    kdir = Path(knowledge_dir)
    if not kdir.exists():
        raise FileNotFoundError(f"Knowledge dir not found: {knowledge_dir}")

    # Load *.md / *.txt (SimpleDirectoryReader handles most encodings)
    docs = SimpleDirectoryReader(input_dir=str(kdir), required_exts=[".md", ".txt",".pdf"]).load_data()
    if not docs:
        raise RuntimeError(f"No knowledge documents found under {knowledge_dir}")

    faiss_index = faiss.IndexFlatIP(dim)
    faiss_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=faiss_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    index.storage_context.persist(persist_dir=persist_dir)
    meta = {
        "domain": "knowledge",
        "count": len(docs),
        "dim": dim,
        "persist_dir": persist_dir,
    }
    ps["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ------------------------------ query -----------------------------------------

def _load_knowledge_index(persist_dir: str):
    ps = _paths(persist_dir)
    if not (ps["faiss"].exists() and ps["docstore"].exists()):
        raise FileNotFoundError(f"No knowledge index in {persist_dir}. Run ensure_knowledge_index first.")
    faiss_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=persist_dir)
    return VectorStoreIndex.from_vector_store(faiss_store, storage_context=storage_context)




def query_knowledge(
    query: str,
    *,
    top_k: int = 3,
    config_path: str = "config/app.yaml",
    persist_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cfg = _cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    persist_dir = persist_dir or str(Path(idx_cfg.get("dir", "var/indexes")) / "knowledge")

    _build_embed_model(cfg)

    index = _load_knowledge_index(persist_dir)
    qe = index.as_query_engine(similarity_top_k=top_k)
    resp = qe.query(query)

    out: List[Dict[str, Any]] = []
    for n in getattr(resp, "source_nodes", []) or []:
        out.append({
            "score": float(getattr(n, "score", 0.0) or 0.0),
            "text": n.get_text(),
            "payload": {"metadata": n.node.metadata if hasattr(n, "node") else {}},
        })
    return out