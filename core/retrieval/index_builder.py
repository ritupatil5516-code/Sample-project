# core/index/index_builder.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding  # or your provider

from core.retrieval.knowledge_ingest import ensure_knowledge_index
from core.retrieval.json_ingest import build_account_index

# configure embeddings once
def _configure_embeddings() -> None:
    # swap to your provider if needed
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# core/index/index_builder.py
from pathlib import Path
from core.retrieval.knowledge_ingest import ensure_knowledge_index
from core.retrieval.json_ingest import build_account_index

INDEX_STORE_DIR = Path("src/api/contextApp/indexesstore")
ACCOUNTS_DATA_DIR = Path("src/api/contextApp/customer_data")      # where the 4 JSONs live
KNOWLEDGE_DATA_DIR = Path("src/api/contextApp/data/knowledge")    # handbook + agreement

def build_all_indexes(account_ids: list[str]):
    print("[BOOT] Building all indexes…")

    # ---- Knowledge
    knowledge_li_dir = INDEX_STORE_DIR / "knowledge" / "llama"
    # optional: start fresh
    knowledge_li_dir.mkdir(parents=True, exist_ok=True)

    ensure_knowledge_index(
        knowledge_dir=KNOWLEDGE_DATA_DIR,
        persist_dir=knowledge_li_dir,
        files=[
            KNOWLEDGE_DATA_DIR / "handbook.md",
            KNOWLEDGE_DATA_DIR / "Apple-Card-Customer-Agreement.pdf",
        ],
        rebuild=True,  # force overwrite
    )

    # ---- Per-account
    for aid in account_ids:
        acc_root = INDEX_STORE_DIR / "accounts" / aid
        acc_li_dir = acc_root / "llama"
        acc_li_dir.mkdir(parents=True, exist_ok=True)

        build_account_index(
            account_id=aid,
            base_dir=ACCOUNTS_DATA_DIR / aid,   # folder containing the 4 jsons
            persist_dir=acc_li_dir,
            rebuild=True,                        # force overwrite
        )

    print("[BOOT] Index build complete.")