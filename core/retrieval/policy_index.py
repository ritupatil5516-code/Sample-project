# core/retrieval/policy_index.py
"""
Policy indexing + query utilities (LlamaIndex + FAISS).

Exports:
- ensure_policy_index(pdf_path, ...)
- query_policy(query, ...)
- get_policy_snippet(capability, ...)

Notes
-----
* Uses OpenAI embeddings by default (configurable via config/app.yaml).
* Persists FAISS+LlamaIndex artifacts under var/indexes/policy/.
* Safe to call ensure_policy_index multiple times (idempotent).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import yaml

# ---- LlamaIndex / FAISS ----
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
    Settings,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Optional: if you later want Qwen embeddings, you can swap here:
# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# from some_qwen_package import QwenEmbedding   # placeholder


# ------------------------------ config helpers --------------------------------

def _cfg(config_path: str = "config/app.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        return {}
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _build_embed_model(cfg: Dict[str, Any]):
    """
    Builds and sets Settings.embed_model from config.
    Default: OpenAI text-embedding-3-large (dim=3072).
    """
    emb = (cfg.get("embeddings") or {})
    provider = (emb.get("provider") or "openai").strip().lower()

    if provider == "qwen":
        # Example placeholder if you later wire Qwen into LlamaIndex
        # base = emb.get("qwen_base_url")
        # model = emb.get("qwen_model", "qwen3-embedding")
        # api_key = os.getenv(emb.get("qwen_api_key_env", "QWEN_API_KEY"), "")
        # Settings.embed_model = QwenEmbedding(api_base=base, api_key=api_key, model=model)
        raise NotImplementedError("Qwen embeddings via LlamaIndex not yet wired.")
    else:
        base = (emb.get("openai_base_url") or "https://api.openai.com/v1").strip()
        model = (emb.get("openai_model") or "text-embedding-3-large").strip()
        key_env = (emb.get("openai_api_key_env") or "OPENAI_API_KEY").strip()
        api_key = (os.getenv(key_env) or "").strip()
        if not api_key:
            raise RuntimeError(f"[policy_index] Missing embedding API key. Set env var {key_env}.")
        # LlamaIndex OpenAIEmbedding reads key from env; base can be passed in via client params
        Settings.embed_model = OpenAIEmbedding(model=model, api_base=base)


def _probe_dim() -> int:
    """
    Infer embedding dimension by embedding a tiny string once.
    (text-embedding-3-large => 3072)
    """
    vec = Settings.embed_model.get_text_embedding("dim_probe")
    return len(vec)


def _policy_paths(persist_dir: str) -> Dict[str, Path]:
    d = Path(persist_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "faiss": d / "faiss.index",
        "docstore": d / "docstore.json",
        "schema": d / "index_store.json",
        "vector": d / "vector_store.json",
        "meta": d / "policy_meta.json",
    }


# ------------------------------ build / ensure --------------------------------

def ensure_policy_index(
    pdf_path: str,
    *,
    config_path: str = "config/app.yaml",
    persist_dir: Optional[str] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build/refresh the FAISS index for the policy PDF.
    Domain persisted at: var/indexes/policy/
    """
    cfg = _cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    persist_dir = persist_dir or str(Path(idx_cfg.get("dir", "var/indexes")) / "policy")

    # 1) set embed model
    _build_embed_model(cfg)
    dim = _probe_dim()

    # 2) paths / short-circuit
    ps = _policy_paths(persist_dir)
    if (not rebuild) and ps["faiss"].exists() and ps["docstore"].exists():
        # already built; return meta
        try:
            return json.loads(ps["meta"].read_text(encoding="utf-8"))
        except Exception:
            pass  # fall through to rebuild

    # 3) load doc
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"Policy PDF not found at {pdf_path}")

    # SimpleDirectoryReader can read a single file:
    docs = SimpleDirectoryReader(input_files=[str(p)]).load_data()
    if not docs:
        raise RuntimeError("Failed to read policy PDF or returned no pages.")

    # 4) build FAISS vector store + index
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=faiss_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    # 5) persist
    index.storage_context.persist(persist_dir=persist_dir)
    meta = {
        "domain": "policy",
        "count": len(docs),
        "dim": dim,
        "persist_dir": persist_dir,
    }
    ps["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ------------------------------ query -----------------------------------------

def _load_policy_index(persist_dir: str):
    """Load a previously persisted policy index."""
    ps = _policy_paths(persist_dir)
    if not (ps["faiss"].exists() and ps["docstore"].exists()):
        raise FileNotFoundError(f"No policy index found in {persist_dir}. Run ensure_policy_index first.")
    # Recreate stores from persisted dir
    faiss_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=faiss_store, persist_dir=persist_dir)
    return VectorStoreIndex.from_vector_store(faiss_store, storage_context=storage_context)


def query_policy(
    query: str,
    *,
    top_k: int = 3,
    config_path: str = "config/app.yaml",
    persist_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cfg = _cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    persist_dir = persist_dir or str(Path(idx_cfg.get("dir", "var/indexes")) / "policy")

    # Ensure embed model is set (for similarity)
    _build_embed_model(cfg)

    index = _load_policy_index(persist_dir)
    qe = index.as_query_engine(similarity_top_k=top_k)
    resp = qe.query(query)

    # LlamaIndex returns a Response object with source_nodes
    out: List[Dict[str, Any]] = []
    for n in getattr(resp, "source_nodes", []) or []:
        out.append({
            "score": float(getattr(n, "score", 0.0) or 0.0),
            "text": n.get_text(),
            "payload": {"metadata": n.node.metadata if hasattr(n, "node") else {}},
        })
    return out


# ------------------------------ optional helper -------------------------------

_CAPABILITY_QUERIES = {
    "grace_period_rules": "Explain the purchase grace period and when interest starts accruing.",
    "dpr_formula": "How is the daily periodic rate applied to balances?",
    "min_payment": "How is the minimum payment calculated?",
    "co_owner": "What is a co-owner and what can they do on the account?",
    "statement_cycle": "What is the statement cycle duration and close date behavior?",
}

def get_policy_snippet(
    capability: str,
    *,
    top_k: int = 2,
    config_path: str = "config/app.yaml",
    persist_dir: Optional[str] = None,
) -> Dict[str, Any]:
    cap = (capability or "").strip().lower()
    q = _CAPABILITY_QUERIES.get(cap)
    if not q:
        return {"error": f"Unknown policy capability '{capability}'", "trace": {"capability": capability}}
    hits = query_policy(q, top_k=top_k, config_path=config_path, persist_dir=persist_dir)
    return {
        "snippets": [{"score": h["score"], "text": h["text"]} for h in hits],
        "trace": {"capability": capability, "query": q, "top_k": top_k, "count": len(hits)},
    }