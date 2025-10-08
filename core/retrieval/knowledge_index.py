# core/retrieval/knowledge_index.py
"""
Knowledge (Handbook + Policy) indexing and semantic retrieval utilities.

Combines multiple reference documents (e.g., handbook.md, Apple Card Agreement)
into a single semantic FAISS index under the 'knowledge' domain.

- ensure_knowledge_index(sources): builds/refreshes the unified FAISS index
- semantic_reference(query): semantic search across all handbook/policy docs
- get_topic_snippet(topic): optional helper to fetch pre-defined conceptual snippets

Dependencies:
  - core/index/faiss_registry.py: Embedder, index_text_file, query_index
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.index.faiss_registry import (
    Embedder,
    index_text_file,
    query_index,
)


# ------------------------------ config helpers ---------------------------------------

def _read_app_cfg(config_path: str = "config/app.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _build_embedder_from_cfg(cfg: Dict[str, Any]) -> Embedder:
    """
    Create an Embedder according to config.embeddings (OpenAI default).
    """
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
        raise RuntimeError(f"[knowledge_index] Missing embedding API key. Set env var {key_env}")

    return Embedder(provider=provider, model=model, api_key=api_key, api_base=base)


# ------------------------------ build index -----------------------------------

def ensure_knowledge_index(
    sources: List[str],
    *,
    config_path: str = "config/app.yaml",
    index_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    chunk_size: int = 700,
    overlap: int = 150,
) -> Dict[str, Any]:
    """
    Build or refresh a unified FAISS index from one or more reference docs.
    Domain name: 'knowledge'
    """
    cfg = _read_app_cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    index_dir = index_dir or idx_cfg.get("dir", "var/indexes")

    emb = embedder or _build_embedder_from_cfg(cfg)

    total_chunks = 0
    indexed_sources = []
    for src in sources:
        p = Path(src)
        if not p.exists():
            print(f"[WARN] Missing knowledge source {src}")
            continue

        meta = index_text_file(
            path=p.as_posix(),
            domain="knowledge",
            embedder=emb,
            index_dir=index_dir,
            max_chars=chunk_size,
            overlap=overlap,
        )
        total_chunks += meta.get("count", 0)
        indexed_sources.append(p.name)

    return {
        "ok": True,
        "count": total_chunks,
        "sources": indexed_sources,
    }


# ------------------------------ semantic retrieval -----------------------------

def semantic_reference(
    query: str,
    *,
    top_k: int = 5,
    config_path: str = "config/app.yaml",
    index_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
) -> Dict[str, Any]:
    """
    Perform a semantic search across the unified knowledge index.
    Returns summarized text chunks for contextual reference.
    """
    if not query:
        return {"error": "Empty query", "trace": {"query": query}}

    cfg = _read_app_cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    index_dir = index_dir or idx_cfg.get("dir", "var/indexes")

    emb = embedder or _build_embedder_from_cfg(cfg)
    hits = query_index(domain="knowledge", query=query, top_k=top_k, index_dir=index_dir, embedder=emb)

    if not hits:
        return {"text": "No relevant handbook or agreement information found.", "trace": {"query": query}}

    results = [h["text"] for h in hits]
    return {
        "text": "\n\n".join(results),
        "trace": {"query": query, "top_k": top_k, "count": len(hits)},
    }


# ------------------------------ optional topic helper -------------------------

_TOPIC_QUERIES = {
    "interest_calculation": "How is interest calculated on credit card balances?",
    "statement_cycle": "What is the duration of a statement cycle and how is it defined?",
    "minimum_payment": "How is the minimum payment amount determined?",
    "co_owner": "What does it mean to be a co-owner on an Apple Card account?",
    "late_fees": "When do late fees apply and how are they assessed?",
}

def get_topic_snippet(
    topic: str,
    *,
    config_path: str = "config/app.yaml",
    index_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Convenience helper used for conceptual questions like:
    'How is interest calculated?' or 'What is a co-owner?'
    """
    key = (topic or "").strip().lower()
    q = _TOPIC_QUERIES.get(key)
    if not q:
        return {"error": f"Unknown knowledge topic '{topic}'", "trace": {"topic": topic}}

    return semantic_reference(q, top_k=top_k, config_path=config_path, index_dir=index_dir, embedder=embedder)


# --- Back-compat shim for legacy 'policy' entry points -----------------------

# Make sure these keys exist so legacy callers still retrieve meaningful text.
# (Add/keep anything you rely on.)
_TOPIC_QUERIES.update({
    "grace_period_rules": "What is the grace period for purchases and when is interest charged?",
    "dpr_formula": "How is the daily periodic rate (APR) applied to balances?",
    "cash_advance_rules": "Rules or fees for cash advances?",
    "fees": "What fees can be charged (late fees, foreign transaction fees, etc.)?",
})

def get_policy_snippet(
    capability: str,
    *,
    config_path: str = "config/app.yaml",
    index_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Backward-compatible alias so older code that imports
    'get_policy_snippet' keeps working. It simply delegates
    to the knowledge topic helper.
    """
    return get_topic_snippet(
        capability,
        config_path=config_path,
        index_dir=index_dir,
        embedder=embedder,
        top_k=top_k,
    )