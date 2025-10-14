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

def build_all_indexes(
    data_root: str | Path = "data",
    index_root: str | Path = "var/indexes",
) -> Dict[str, Any]:
    """
    Builds:
    - Knowledge index: handbook.md + Apple-Card-Customer-Agreement.pdf
    - One FAISS index per account (4 JSONs per account)
    """
    _configure_embeddings()
    data_root  = Path(data_root).resolve()
    index_root = Path(index_root).resolve()
    index_root.mkdir(parents=True, exist_ok=True)

    print("[BOOT] Building all indexes…")

    # 1) Knowledge
    try:
        knowledge_dir = data_root / "knowledge"
        agreement_pdf = data_root / "agreement" / "Apple-Card-Customer-Agreement.pdf"
        handbook_md   = knowledge_dir / "handbook.md"
        knowledge_out = index_root / "knowledge"
        meta_k = ensure_knowledge_index(
            knowledge_dir=knowledge_dir,
            persist_dir=knowledge_out,
            files=[handbook_md, agreement_pdf],  # <— LIST fixes your first error
        )
        print(f"[INDEX] Knowledge: {meta_k['count']} docs → {meta_k['persist_dir']}")
    except Exception as e:
        print(f"[WARN] Knowledge index build failed: {e}")

    # 2) Accounts
    # assume folder structure: data/customer_data/<account_id>/{transactions.json, payments.json, statements.json, account_summary.json}
    cust_dir = data_root / "customer_data"
    if cust_dir.exists():
        for account_dir in sorted([p for p in cust_dir.iterdir() if p.is_dir()]):
            aid = account_dir.name
            try:
                out_dir = index_root / "accounts" / aid
                meta_a = build_account_index(account_id=aid, base_dir=account_dir, persist_dir=out_dir)
                print(f"[INDEX] Account {aid}: {meta_a['count']} docs → {meta_a['persist_dir']}")
            except Exception as e:
                print(f"[WARN] Failed to build index for {aid}: {e}")

    print("[BOOT] Index build complete.")
    return {"index_root": str(index_root)}