# core/retrieval/policy_index.py
"""
Policy indexing + query utilities.

- ensure_policy_index(pdf_path, ...): builds a FAISS index from the Apple Card Agreement (PDF/text)
- query_policy(query, ...): semantic search over the policy index
- get_policy_snippet(capability, ...): optional helper to fetch common policy snippets by capability

Relies on: core/index/faiss_registry.py (Embedder, index_text_file, query_index)
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


# ------------------------------ helpers ---------------------------------------

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
        raise RuntimeError(f"[policy_index] Missing embedding API key. Set env var {key_env}")

    return Embedder(provider=provider, model=model, api_key=api_key, api_base=base)


# ------------------------------ build index -----------------------------------

def ensure_policy_index(
    pdf_path: str,
    *,
    config_path: str = "config/app.yaml",
    index_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    chunk_size: int = 600,
    overlap: int = 150,
) -> Dict[str, Any]:
    """
    Build/refresh the FAISS index for the Apple Card Agreement (policy).
    Uses domain name: 'policy'
    """
    cfg = _read_app_cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    index_dir = index_dir or idx_cfg.get("dir", "var/indexes")

    emb = embedder or _build_embedder_from_cfg(cfg)

    # index_text_file handles .pdf reading via PyPDF2 (as implemented in faiss_registry)
    meta = index_text_file(
        path=pdf_path,
        domain="policy",
        embedder=emb,
        index_dir=index_dir,
        max_chars=chunk_size,
        overlap=overlap,
    )
    return meta


# ------------------------------ query index -----------------------------------

def query_policy(
    query: str,
    *,
    top_k: int = 3,
    config_path: str = "config/app.yaml",
    index_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic query over the policy index (domain='policy').
    Returns list of {score, text, payload, idx, meta}.
    """
    cfg = _read_app_cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    index_dir = index_dir or idx_cfg.get("dir", "var/indexes")

    emb = embedder or _build_embedder_from_cfg(cfg)
    hits = query_index(domain="policy", query=query, top_k=top_k, index_dir=index_dir, embedder=emb)
    return hits


# ------------------------------ optional capability helper --------------------

_CAPABILITY_QUERIES = {
    # Add/adjust these mappings to tune retrieval prompts as needed.
    "grace_period_rules": "What is the grace period for Apple Card purchases and when is interest charged?",
    "dpr_formula": "How is the daily periodic rate (APR) applied to Apple Card balances?",
    "cash_advance_rules": "Are there rules or fees for cash advances with Apple Card?",
    "fees": "What fees can be charged on Apple Card (e.g., late fees, foreign transaction fees)?",
}

def get_policy_snippet(
    capability: str,
    *,
    config_path: str = "config/app.yaml",
    index_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    top_k: int = 2,
) -> Dict[str, Any]:
    """
    Convenience helper used by the orchestrator 'policy' domain.
    Maps a capability to a retrieval query and returns top snippets.
    """
    cap = (capability or "").strip().lower()
    q = _CAPABILITY_QUERIES.get(cap)
    if not q:
        return {"error": f"Unknown policy capability '{capability}'", "trace": {"capability": capability}}

    hits = query_policy(q, top_k=top_k, config_path=config_path, index_dir=index_dir, embedder=embedder)
    return {
        "snippets": [{"score": h["score"], "text": h["text"]} for h in hits],
        "trace": {"capability": capability, "query": q, "top_k": top_k, "count": len(hits)},
    }