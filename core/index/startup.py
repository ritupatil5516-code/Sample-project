# core/index/startup.py
"""
Startup index builder.

Builds:
  1) Domain semantic indexes (JSON): transactions, payments, statements, account_summary
  2) Temporal indexes (JSON + timestamps): transactions, payments, statements
  3) Semantic knowledge indexes (text/PDF): handbook.md, Apple Card Agreement

Dependencies:
  - core/index/faiss_registry.py: index_json_file, index_text_file, Embedder
  - core/index/temporal_registry.py: ensure_temporal_from_json
  - core/retrieval/policy_index.py: ensure_policy_index
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.index.faiss_registry import index_json_file, index_text_file, Embedder
from core.index.temporal_registry import ensure_temporal_from_json
from core.retrieval.policy_index import ensure_policy_index


# ------------------------------ helpers ---------------------------------------

def _index_exists(index_dir: str, domain: str, *, temporal: bool = False) -> bool:
    base = f"{domain}_temporal" if temporal else domain
    d = Path(index_dir)
    return (d / f"{base}.index").exists() and (d / f"{base}_meta.json").exists()


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


def _as_bool(val: Any, default: bool = True) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _build_embedder_from_cfg(cfg: Dict[str, Any]) -> Embedder:
    """
    Create a single Embedder instance shared by all index builds/queries,
    honoring your config/app.yaml 'embeddings' section.
    """
    emb_cfg = cfg.get("embeddings", {}) or {}
    provider = (emb_cfg.get("provider") or "openai").strip().lower()

    if provider == "qwen":
        base = (emb_cfg.get("qwen_base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        model = (emb_cfg.get("qwen_model") or "qwen3-embedding").strip()
        key_env = (emb_cfg.get("qwen_api_key_env") or "QWEN_API_KEY").strip()
    else:
        # default openai
        base = (emb_cfg.get("openai_base_url") or "https://api.openai.com/v1").strip()
        model = (emb_cfg.get("openai_model") or "text-embedding-3-large").strip()
        key_env = (emb_cfg.get("openai_api_key_env") or "OPENAI_API_KEY").strip()

    api_key = (os.getenv(key_env) or "").strip()
    if not api_key:
        raise RuntimeError(f"Missing embedding API key. Set env var {key_env}")

    return Embedder(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=base,
    )


# ---------------------------- main entrypoint ---------------------------------

def build_on_startup(config_path: str = "config/app.yaml") -> Dict[str, Any]:
    """
    Build or verify all indexes required by the copilot. Safe to call multiple times.
    Honors:
      - indexes.dir
      - indexes.rebuild_on_startup
      - embeddings.* (provider/model/base/key_env)
    """
    print("[INIT] Starting index build…")

    cfg = _read_app_cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    index_dir = idx_cfg.get("dir", "var/indexes")
    rebuild = _as_bool(idx_cfg.get("rebuild_on_startup"), False)

    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # Create ONE embedder for everything (consistent key/base/model)
    try:
        embedder = _build_embedder_from_cfg(cfg)
        print(f"[INIT] Embedder => provider={embedder.provider}, model={embedder.model}, base={embedder.api_base}")
    except Exception as e:
        # Fail fast with a clear message instead of cryptic header errors.
        raise

    # -------------------------------------------------------------------------
    # 1) Domain JSON → semantic FAISS
    # -------------------------------------------------------------------------
    domain_files = {
        "transactions": "data/folder/transactions.json",
        "payments": "data/folder/payments.json",
        "statements": "data/folder/statements.json",
        "account_summary": "data/folder/account_summary.json",
    }

    domain_index_results: Dict[str, Any] = {}
    for domain, path in domain_files.items():
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Missing {path}, skipping {domain} semantic index")
            continue

        if (not rebuild) and _index_exists(index_dir, domain, temporal=False):
            # Already built; load meta for summary
            meta_path = Path(index_dir) / f"{domain}_meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            domain_index_results[domain] = {"ok": True, "meta": meta, "skipped": True}
            print(f"[SKIP] Semantic index exists for {domain}")
            continue

        try:
            meta = index_json_file(
                path=p.as_posix(),
                domain=domain,
                embedder=embedder,          # <<< use the configured embedder
                index_dir=index_dir,
            )
            domain_index_results[domain] = {"ok": True, "meta": meta}
            print(f"[OK] Semantic index built for {domain} ({meta.get('count')} rows)")
        except Exception as e:
            domain_index_results[domain] = {"ok": False, "error": str(e)}
            print(f"[ERR] Failed semantic index for {domain}: {e}")

    # -------------------------------------------------------------------------
    # 2) Temporal JSON → time-aware FAISS (recency-friendly)
    # -------------------------------------------------------------------------
    temporal_results: Dict[str, Any] = {}
    temporal_specs = [
        ("transactions", "data/folder/transactions.json", ["transactionDateTime", "postedDateTime", "date"]),
        ("payments", "data/folder/payments.json", ["paymentDateTime", "paymentPostedDateTime", "date"]),
        ("statements", "data/folder/statements.json", ["closingDateTime", "openingDateTime", "period"]),
    ]

    for domain, path, candidates in temporal_specs:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Missing {path}, skipping {domain} temporal index")
            continue

        if (not rebuild) and _index_exists(index_dir, domain, temporal=True):
            meta_path = Path(index_dir) / f"{domain}_temporal_meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            temporal_results[domain] = {"ok": True, "meta": meta, "skipped": True}
            print(f"[SKIP] Temporal index exists for {domain}")
            continue

        try:
            # ensure_temporal_from_json manages its own existence via 'rebuild'
            meta = ensure_temporal_from_json(
                domain=domain,
                json_path=p.as_posix(),
                index_dir=index_dir,
                time_field_candidates=candidates,
                rebuild=True if rebuild else False,
            )
            temporal_results[domain] = {"ok": True, "meta": meta}
            print(f"[OK] Temporal index built for {domain} ({meta.get('count')} rows)")
        except Exception as e:
            temporal_results[domain] = {"ok": False, "error": str(e)}
            print(f"[ERR] Failed temporal index for {domain}: {e}")

    # -------------------------------------------------------------------------
    # 3) Semantic knowledge (text/PDF)
    # -------------------------------------------------------------------------
    knowledge_results: Dict[str, Any] = {}

    # 3a) Handbook (merged MD)
    handbook = Path("data/knowledge/handbook.md")
    if handbook.exists():
        if (not rebuild) and _index_exists(index_dir, "knowledge", temporal=False):
            meta_path = Path(index_dir) / "knowledge_meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            knowledge_results["handbook"] = {"ok": True, "meta": meta, "skipped": True}
            print("[SKIP] Knowledge index already present")
        else:
            try:
                meta = index_text_file(
                    path=handbook.as_posix(),
                    domain="knowledge",
                    embedder=embedder,       # <<< use the configured embedder
                    index_dir=index_dir,
                )
                knowledge_results["handbook"] = {"ok": True, "meta": meta}
                print(f"[OK] Knowledge index (handbook) built with {meta.get('count')} chunks")
            except Exception as e:
                knowledge_results["handbook"] = {"ok": False, "error": str(e)}
                print(f"[ERR] Failed knowledge index (handbook): {e}")
    else:
        print("[WARN] data/knowledge/handbook.md missing; skipping knowledge index")

    # 3b) Apple Card Agreement (PDF → policy index)
    policy_pdf = Path("data/agreement/Apple-Card-Customer-Agreement.pdf")
    if policy_pdf.exists():
        # ensure_policy_index internally extracts text and builds FAISS under 'policy' domain.
        # If your ensure_policy_index can accept an embedder, pass it here; otherwise it will
        # use its own Embedder (make sure that code also honors config/env).
        try:
            meta = ensure_policy_index(policy_pdf.as_posix())
            knowledge_results["policy_pdf"] = {"ok": True, "meta": meta}
            print("[OK] Policy index built from Apple Card Agreement PDF")
        except Exception as e:
            knowledge_results["policy_pdf"] = {"ok": False, "error": str(e)}
            print(f"[ERR] Failed policy PDF index: {e}")
    else:
        print("[WARN] Apple Card Agreement PDF missing; skipping policy index")

    # -------------------------------------------------------------------------
    # 4) Summary
    # -------------------------------------------------------------------------
    summary = {
        "index_dir": index_dir,
        "rebuild_on_startup": rebuild,
        "domains_semantic": domain_index_results,
        "domains_temporal": temporal_results,
        "knowledge": knowledge_results,
    }
    print("[INIT] Index building complete.")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build_on_startup()