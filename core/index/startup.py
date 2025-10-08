# core/index/startup.py
"""
Startup index builder.

Builds:
  1) Domain semantic indexes (JSON): transactions, payments, statements, account_summary
  2) Temporal indexes (JSON + timestamps): transactions, payments, statements
  3) Unified knowledge index (text/PDF): handbook.md + Apple Card Agreement

Dependencies:
  - core/index/faiss_registry.py: index_json_file, Embedder
  - core/index/temporal_registry.py: ensure_temporal_from_json
  - core/retrieval/knowledge_index.py: ensure_knowledge_index
"""

from __future__ import annotations
import os, json, yaml
from pathlib import Path
from typing import Any, Dict

from core.index.faiss_registry import index_json_file, Embedder
from core.index.temporal_registry import ensure_temporal_from_json
from core.retrieval.knowledge_index import ensure_knowledge_index


def _read_app_cfg(config_path="config/app.yaml") -> Dict[str, Any]:
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
    if isinstance(val, bool): return val
    if val is None: return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"): return True
    if s in ("0", "false", "no", "n", "off"): return False
    return default


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
        raise RuntimeError(f"Missing embedding API key. Set env var {key_env}")
    return Embedder(provider=provider, model=model, api_key=api_key, api_base=base)


def build_on_startup(config_path="config/app.yaml") -> Dict[str, Any]:
    print("[INIT] Starting index build…")

    cfg = _read_app_cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    index_dir = idx_cfg.get("dir", "var/indexes")
    rebuild = _as_bool(idx_cfg.get("rebuild_on_startup"), False)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    embedder = _build_embedder_from_cfg(cfg)
    print(f"[INIT] Embedder => provider={embedder.provider}, model={embedder.model}")

    # 1️⃣ Domain semantic indexes
    domain_files = {
        "transactions": "data/folder/transactions.json",
        "payments": "data/folder/payments.json",
        "statements": "data/folder/statements.json",
        "account_summary": "data/folder/account_summary.json",
    }
    for domain, path in domain_files.items():
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Missing {path}, skipping {domain}")
            continue
        index_json_file(path=p.as_posix(), domain=domain, embedder=embedder, index_dir=index_dir)
        print(f"[OK] Built semantic index for {domain}")

    # 2️⃣ Temporal indexes
    for domain, path, cands in [
        ("transactions", "data/folder/transactions.json", ["transactionDateTime", "postedDateTime", "date"]),
        ("payments", "data/folder/payments.json", ["paymentDateTime", "paymentPostedDateTime", "date"]),
        ("statements", "data/folder/statements.json", ["closingDateTime", "openingDateTime", "period"]),
    ]:
        p = Path(path)
        if not p.exists(): continue
        ensure_temporal_from_json(domain=domain, json_path=p.as_posix(), index_dir=index_dir, time_field_candidates=cands, rebuild=rebuild)
        print(f"[OK] Built temporal index for {domain}")

    # 3️⃣ Knowledge Index
    handbook = Path("data/knowledge/handbook.md")
    agreement = Path("data/agreement/Apple-Card-Customer-Agreement.pdf")
    sources = [p.as_posix() for p in [handbook, agreement] if p.exists()]
    if sources:
        meta = ensure_knowledge_index(sources=sources, config_path=config_path, embedder=embedder)
        print(f"[OK] Knowledge index built from {meta.get('count')} text chunks.")
    else:
        print("[WARN] No knowledge sources found; skipping knowledge index")

    print("[INIT] Index building complete.")
    return {"index_dir": index_dir, "sources": sources}


if __name__ == "__main__":
    build_on_startup()