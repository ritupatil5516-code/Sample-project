# settings.py (repo root)
from pathlib import Path
import os, json, yaml
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent  # e.g., /app

def _choose_dir(env_name: str, default_rel: str) -> Path:
    raw = os.getenv(env_name)
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p = p.expanduser().resolve()
        if p.exists():
            return p
        print(f"[settings] WARNING: {env_name}='{raw}' missing; falling back to {default_rel}", flush=True)
    return (PROJECT_ROOT / default_rel).expanduser().resolve()

DATA_DIR           = _choose_dir("DATA_DIR",          "src/api/contextApp/data")
CONTEXT_PACK_DIR   = _choose_dir("CONTEXT_PACK_DIR",  "src/api/contextApp/context/packs")
CONFIG_DIR         = _choose_dir("CONFIG_DIR",        "src/api/contextApp/config")
INDEX_STORE_DIR    = _choose_dir("INDEX_STORE_DIR",   "src/api/contextApp/indexes_store")
ACCOUNTS_DATA_DIR  = _choose_dir("ACCOUNTS_DATA_DIR", "src/api/contextApp/data/customer_data")
KNOWLEDGE_DATA_DIR = _choose_dir("KNOWLEDGE_DATA_DIR","src/api/contextApp/data/knowledge")

CONFIG_FILE = (Path(os.getenv("CONFIG_FILE", "")) if os.getenv("CONFIG_FILE") else CONFIG_DIR / "app.yaml")
CONFIG_FILE = (CONFIG_FILE if CONFIG_FILE.is_absolute() else PROJECT_ROOT / CONFIG_FILE).resolve()
if not CONFIG_FILE.exists():
    # final safety: always fall back to repo default
    alt = (CONFIG_DIR / "app.yaml").resolve()
    print(f"[settings] WARNING: CONFIG_FILE missing at {CONFIG_FILE}; fallback to {alt}", flush=True)
    CONFIG_FILE = alt

CORE_FILE = (CONTEXT_PACK_DIR / "core.yaml").resolve()

# Optional: one-time debug
print(f"[settings] PROJECT_ROOT={PROJECT_ROOT}", flush=True)
for k, v in {
    "CONFIG_DIR": CONFIG_DIR, "CONFIG_FILE": CONFIG_FILE,
    "CONTEXT_PACK_DIR": CONTEXT_PACK_DIR, "DATA_DIR": DATA_DIR
}.items():
    print(f"[settings] {k}={v} exists={v.exists()}", flush=True)
print(f"[settings] ENV CONFIG_DIR={os.getenv('CONFIG_DIR')} CONFIG_FILE={os.getenv('CONFIG_FILE')}", flush=True)
