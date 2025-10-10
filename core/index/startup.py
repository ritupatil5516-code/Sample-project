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
  - core/retrieval/knowledge_faiss.py: ensure_knowledge_index
"""

from __future__ import annotations
import os, json, yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.index.faiss_registry import index_json_file, Embedder
from core.index.temporal_registry import ensure_temporal_from_json
from core.retrieval.knowledge_faiss import ensure_knowledge_index


# ------------------------------
# Path resolution (no parents[])
# ------------------------------

def _resolve_index_dir() -> Path:
    """
    Priority:
      1) COPILOT_INDEX_DIR (env)
      2) ./var/indexes (if folder exists)
      3) /tmp/copilot_indexes (AWS/container-safe fallback)
    """
    env_val = os.getenv("COPILOT_INDEX_DIR")
    if env_val:
        p = Path(env_val).resolve()
    else:
        local_default = Path("var/indexes").resolve()
        if local_default.exists():
            p = local_default
        else:
            p = Path("/tmp/copilot_indexes").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_data_dir() -> Path:
    """
    Where your static JSON files live.
    Priority:
      1) COPILOT_DATA_DIR (env)
      2) ./data/folder (your current layout)
      3) ./data        (common alternative)
      4) /app/data     (typical in containers)
      5) /tmp/copilot_data (last-resort, created)
    """
    candidates: List[str] = [
        os.getenv("COPILOT_DATA_DIR", "").strip(),
        str(Path("data/folder").resolve()),
        str(Path("data").resolve()),
        "/app/data",
        "/tmp/copilot_data",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c).resolve()
        if p.exists():
            # ensure exists & return
            p.mkdir(parents=True, exist_ok=True)
            return p
    # Fallback create
    p = Path("/tmp/copilot_data").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def _log_paths(index_dir: Path, data_dir: Path) -> None:
    print("[PATHS]")
    print(f"  cwd              = {Path.cwd()}")
    print(f"  index_dir        = {index_dir}")
    print(f"  data_dir         = {data_dir}")
    print(f"  COPILOT_FORCE_REBUILD = {os.getenv('COPILOT_FORCE_REBUILD', '0')}")


def _exists(p: Path, label: str) -> bool:
    ok = p.exists()
    if not ok:
        print(f"[WARN] Missing {label}: {p} (skipping)")
    return ok


def _domain_specs(data_dir: Path) -> Tuple[Dict[str, Path], list[tuple[str, Path, list[str]]]]:
    """
    Returns:
      - mapping for semantic JSON indexes
      - list for temporal JSON indexes (with candidate time fields)
    """
    sem = {
        "transactions": data_dir / "transactions.json",
        "payments": data_dir / "payments.json",
        "statements": data_dir / "statements.json",
        "account_summary": data_dir / "account_summary.json",
    }
    temporal = [
        ("transactions", sem["transactions"], ["transactionDateTime", "postedDateTime", "date"]),
        ("payments",     sem["payments"],     ["paymentDateTime", "paymentPostedDateTime", "date"]),
        ("statements",   sem["statements"],   ["closingDateTime", "openingDateTime", "period"]),
    ]
    return sem, temporal


def build_on_startup(config_path="config/app.yaml") -> Dict[str, Any]:
    print("[INIT] Starting index build…")

    # Resolve paths first (CWD-proof)
    index_dir = _resolve_index_dir()
    data_dir = _resolve_data_dir()
    _log_paths(index_dir, data_dir)

    # Config & flags
    cfg = _read_app_cfg(config_path)
    idx_cfg = cfg.get("indexes") or {}
    # Allow config to override index_dir if explicitly set there
    index_dir_cfg = (idx_cfg.get("dir") or "").strip()
    if index_dir_cfg:
        index_dir = Path(index_dir_cfg).resolve()
        index_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Overriding index_dir from config: {index_dir}")

    rebuild_cfg = _as_bool(idx_cfg.get("rebuild_on_startup"), False)
    force_rebuild_env = os.getenv("COPILOT_FORCE_REBUILD", "0") == "1"
    rebuild = rebuild_cfg or force_rebuild_env

    ready_marker = index_dir / ".ready"

    embedder = _build_embedder_from_cfg(cfg)
    print(f"[INIT] Embedder => provider={embedder.provider}, model={embedder.model}")

    # If indexes already built and rebuild is false, skip heavy work
    if ready_marker.exists() and not rebuild:
        print(f"[INIT] Indexes already present at {index_dir}; skipping rebuild.")
        return {"index_dir": str(index_dir), "sources": []}

    # 1️⃣ Domain semantic indexes
    semantic_map, temporal_specs = _domain_specs(data_dir)
    for domain, p in semantic_map.items():
        if not _exists(p, f"{domain} json"):
            continue
        index_json_file(path=p.as_posix(), domain=domain, embedder=embedder, index_dir=str(index_dir))
        print(f"[OK] Built semantic index for {domain}")

    # 2️⃣ Temporal indexes
    for domain, p, cands in temporal_specs:
        if not _exists(p, f"{domain} json"):
            continue
        ensure_temporal_from_json(
            domain=domain,
            json_path=p.as_posix(),
            index_dir=str(index_dir),
            time_field_candidates=cands,
            rebuild=rebuild,
        )
        print(f"[OK] Built temporal index for {domain}")

    # 3️⃣ Knowledge Index
    # Default layout: data/knowledge + data/agreement (relative to repo)
    # Also try within data_dir if users placed files alongside JSONs.
    sources: list[str] = []
    candidates = [
        Path("data/knowledge/handbook.md"),
        Path("data/agreement/Apple-Card-Customer-Agreement.pdf"),
        data_dir / "knowledge" / "handbook.md",
        data_dir / "agreement" / "Apple-Card-Customer-Agreement.pdf",
    ]
    for cp in candidates:
        if cp and cp.exists():
            sources.append(cp.as_posix())

    if sources:
        # Your existing API expects a directory root; keep consistent with your current call
        # If your function signature differs, adjust here.
        meta = ensure_knowledge_index("data/knowledge" if Path("data/knowledge").exists() else str(data_dir / "knowledge"))
        count = meta.get("count") if isinstance(meta, dict) else "?"
        print(f"[OK] Knowledge index built. chunks={count}")
    else:
        print("[WARN] No knowledge sources found; skipping knowledge index")

    ready_marker.touch()
    print("[INIT] Index building complete.")
    return {"index_dir": str(index_dir), "sources": sources}


if __name__ == "__main__":
    build_on_startup()