# core/retrieval/json_ingest.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import json


def _embed_dim() -> int:
    return int(getattr(Settings.embed_model, "embed_dim", 1536))


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON file that may be a list or an object.
    Returns a list of dict rows.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def build_account_index(
    account_id: str,
    base_dir: str | Path,          # e.g., data/customer_data/<account_id>
    persist_dir: str | Path,       # e.g., var/indexes/accounts/<account_id>
) -> Dict[str, Any]:
    """
    Builds a FAISS index for a single account by merging its 4 JSON files
    (transactions, payments, statements, account_summary) into one vector index.

    The raw JSON row is stored as text; helpful fields are mirrored in metadata
    for filtering / debugging.
    """
    base_dir = Path(base_dir).resolve()
    persist_dir = Path(persist_dir).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "transactions": base_dir / "transactions.json",
        "payments": base_dir / "payments.json",
        "statements": base_dir / "statements.json",
        "account_summary": base_dir / "account_summary.json",
    }

    docs: List[Document] = []
    for domain, p in files.items():
        if not p.exists():
            continue
        rows = _load_json(p)
        for r in rows:
            # Keep full row as text and add a few useful metadata fields
            docs.append(
                Document(
                    text=json.dumps(r, ensure_ascii=False),
                    metadata={
                        "account_id": account_id,
                        "domain": domain,
                        # common shortcuts (exist or not depending on file)
                        "merchantName": r.get("merchantName"),
                        "transactionType": r.get("transactionType"),
                        "displayTransactionType": r.get("displayTransactionType"),
                        "amount": r.get("amount"),
                        "period": r.get("period"),
                        "postedDateTime": r.get("postedDateTime"),
                        "transactionDateTime": r.get("transactionDateTime"),
                        "closingDateTime": r.get("closingDateTime"),
                    },
                )
            )

    dim = _embed_dim()
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))

    index = VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist(persist_dir=str(persist_dir))

    return {"count": len(docs), "persist_dir": str(persist_dir), "dim": dim}