# core/index/temporal_registry.py
"""
Temporal index builder for JSON rows.

Creates a separate FAISS index per domain with name: <domain>_temporal
The text includes the detected time field and ISO timestamp to make recency-aware
queries easier (e.g., “last transaction”).

Public:
- ensure_temporal_from_json(domain, json_path, index_dir, time_field_candidates, rebuild=False, embedder=None)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from core.index.faiss_registry import Embedder, _read_json_objects, _paths, _build_faiss

import numpy as np
import faiss


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
        raise RuntimeError(f"[temporal_registry] Missing embedding API key. Set env var {key_env}")

    return Embedder(provider=provider, model=model, api_key=api_key, api_base=base)


def _parse_timestamp(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """
    Try candidate keys to extract a timestamp-like field.
    Return normalized ISO-8601 string if possible, else None.
    """
    for key in candidates:
        v = row.get(key)
        if not v:
            continue
        s = str(v)
        # Accept raw ISO strings or 'YYYY-MM'/'YYYY-MM-DD' etc.
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",  # 2025-09-30T23:59:59Z
            "%Y-%m-%dT%H:%M:%S",   # 2025-09-30T23:59:59
            "%Y-%m-%d",            # 2025-09-30
            "%Y-%m",               # 2025-09
        ):
            try:
                dt = datetime.strptime(s, fmt)
                # Assume UTC if no tz; keep Z for consistency
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
        # Last resort: if looks like YYYY-MM or YYYY-MM-DD, keep as-is
        if len(s) in (7, 10) and s[4] == "-":
            return s
    return None


def _row_to_temporal_text(row: Dict[str, Any], iso_ts: Optional[str]) -> str:
    """
    Render a JSON row to a short text with the timestamp up front to bias recency queries.
    """
    parts = []
    if iso_ts:
        parts.append(f"__ts__={iso_ts}")
    # Compact key=val pairs similar to _default_row_text
    for k, v in row.items():
        try:
            if isinstance(v, (dict, list)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v)
        except Exception:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return " | ".join(parts)


def _write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ------------------------------ builder ---------------------------------------

def ensure_temporal_from_json(
    *,
    domain: str,
    json_path: str,
    index_dir: str = "var/indexes",
    time_field_candidates: Optional[List[str]] = None,
    rebuild: bool = False,
    embedder: Optional[Embedder] = None,
    config_path: str = "config/app.yaml",
) -> Dict[str, Any]:
    """
    Build/refresh a temporal FAISS index with domain name: '<domain>_temporal'.
    Each row is embedded as text prefixed with its normalized timestamp.
    Also writes parallel rows/texts/meta files similar to semantic indexing.

    Returns meta dict with:
      - domain: "<domain>_temporal"
      - count, dim, provider, model
      - time_field: the candidate key chosen (if any)
    """
    dom_name = f"{domain}_temporal"
    ps = _paths(index_dir, dom_name)

    if (not rebuild) and ps["index"].exists() and ps["meta"].exists():
        # Already built → return meta
        try:
            return json.loads(ps["meta"].read_text(encoding="utf-8"))
        except Exception:
            pass  # fall through to rebuild

    rows = _read_json_objects(Path(json_path))
    if not rows:
        raise ValueError(f"[temporal_registry] No rows found in {json_path}")

    candidates = time_field_candidates or ["date", "timestamp", "transactionDateTime", "paymentDateTime", "closingDateTime", "period"]

    # detect chosen time field by first successful parse
    chosen_field: Optional[str] = None
    iso_list: List[Optional[str]] = []
    for r in rows:
        iso = _parse_timestamp(r, candidates)
        iso_list.append(iso)
        if (not chosen_field) and iso:
            # try to figure which key matched
            for k in candidates:
                if r.get(k):
                    chosen_field = k
                    break

    # Render texts
    texts = [_row_to_temporal_text(r, iso) for r, iso in zip(rows, iso_list)]

    # Build or use embedder
    emb = embedder or _build_embedder_from_cfg(_read_app_cfg(config_path))
    vecs = emb.embed(texts)  # normalized vectors

    # Build FAISS index
    index = _build_faiss(vecs, metric="ip")
    faiss.write_index(index, str(ps["index"]))

    # Persist parallel artifacts
    _write_jsonl(ps["rows"], rows)
    _write_jsonl(ps["texts"], [{"text": t} for t in texts])

    meta = {
        "domain": dom_name,
        "source": str(json_path),
        "count": len(rows),
        "dim": int(vecs.shape[1]),
        "metric": "ip",
        "provider": emb.provider,
        "model": emb.model,
        "created_at": int(datetime.utcnow().timestamp()),
        "type": "temporal-json",
        "time_field": chosen_field,
    }
    ps["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta