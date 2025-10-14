# index_builder.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import faiss
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore

# ---------- Paths (adjust if yours differ) ----------
ROOT = Path(__file__).resolve().parents[3]   # .../src/api/contextApp
DATA_DIR = ROOT / "customer_data"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
INDEX_ROOT = ROOT / "indexesstore"           # persist here

# ---------- Embedding dim helper ----------
def _embed_dim(model: str | None = None) -> int:
    """
    Keep in sync with the embedding model you set in Settings.
    OpenAI text-embedding-3-large -> 3072, small -> 1536.
    """
    return 3072

# ---------- JSON helpers ----------
def _read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _iter_account_docs(account_id: str) -> List[Document]:
    """
    Flatten the 4 account JSONs into LlamaIndex Documents.
    One Document per row (transactions/payments/statements)
    and one Document for account_summary.
    """
    base = DATA_DIR / account_id
    tx = base / "transactions.json"
    py = base / "payments.json"
    st = base / "statements.json"
    ac = base / "account_summary.json"

    docs: List[Document] = []
    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)

    def _docify(row: Dict[str, Any], domain: str) -> List[Document]:
        text = json.dumps(row, ensure_ascii=False)
        # let LlamaIndex chunk it to keep similarity higher granularity
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        out: List[Document] = []
        for nd in nodes:
            out.append(
                Document(
                    text=nd.get_content(),
                    metadata={
                        "account_id": account_id,
                        "domain": domain,
                        # common shortcuts that help in retrieval filters/snippets
                        "merchantName": row.get("merchantName"),
                        "description": row.get("description"),
                        "amount": row.get("amount"),
                        "postedDateTime": row.get("postedDateTime"),
                        "transactionDateTime": row.get("transactionDateTime"),
                        "paymentPostedDateTime": row.get("paymentPostedDateTime"),
                        "period": row.get("period"),
                    },
                )
            )
        return out

    if tx.exists():
        for r in _read_json(tx) or []:
            docs.extend(_docify(r, "transactions"))

    if py.exists():
        for r in _read_json(py) or []:
            docs.extend(_docify(r, "payments"))

    if st.exists():
        for r in _read_json(st) or []:
            docs.extend(_docify(r, "statements"))

    if ac.exists():
        row = _read_json(ac) or {}
        # keep one document for summary (also chunked)
        docs.extend(_docify(row, "account_summary"))

    return docs

def build_account_index(
    account_id: str,
    base_dir: Path | None = None,
    persist_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Build FAISS index for a single account (all 4 JSONs together).
    """
    base_dir = base_dir or (DATA_DIR / account_id)
    persist_dir = persist_dir or (INDEX_ROOT / "accounts" / account_id / "llama")
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = _iter_account_docs(account_id)
    if not docs:
        return {"count": 0, "persist_dir": str(persist_dir), "dim": _embed_dim()}

    dim = _embed_dim()
    faiss_index = faiss.IndexFlatIP(dim)                 # cosine via inner-product
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))
    VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist(persist_dir=str(persist_dir))
    return {"count": len(docs), "persist_dir": str(persist_dir), "dim": dim}

def ensure_knowledge_index(
    knowledge_dir: Path | None = None,
    persist_dir: Path | None = None,
    files: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """
    Build FAISS index over handbook + agreement (or provided files list).
    """
    from llama_index.core import SimpleDirectoryReader  # import here to keep deps light

    knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
    persist_dir = persist_dir or (INDEX_ROOT / "knowledge" / "llama")
    persist_dir.mkdir(parents=True, exist_ok=True)

    if files is None:
        # default two files
        files = [
            knowledge_dir / "handbook.md",
            knowledge_dir / "Apple-Card-Customer-Agreement.pdf",
        ]

    input_files = [str(p) for p in files if Path(p).exists()]
    if not input_files:
        return {"count": 0, "persist_dir": str(persist_dir), "dim": _embed_dim()}

    reader = SimpleDirectoryReader(input_files=input_files)
    docs = reader.load_data()

    dim = _embed_dim()
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))
    VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist(persist_dir=str(persist_dir))
    return {"count": len(docs), "persist_dir": str(persist_dir), "dim": dim}

# ---------- loaders used by RAG ----------
def load_account_index(persist_dir: str | Path):
    p = Path(persist_dir)
    vector_store = FaissVectorStore.from_persist_dir(str(p))
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(p))
    return load_index_from_storage(storage)

def load_knowledge_index(persist_dir: str | Path):
    p = Path(persist_dir)
    vector_store = FaissVectorStore.from_persist_dir(str(p))
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(p))
    return load_index_from_storage(storage)

# ---------- convenience: build all ----------
def ensure_all_indexes(account_ids: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"knowledge": ensure_knowledge_index()}
    for aid in account_ids:
        out[aid] = build_account_index(aid)
    return out

if __name__ == "__main__":
    # Example: build everything on startup for a known account
    # (adjust account id or pass a list)
    aid = "3b1ba69f-8c69-4e7c-94e2-2e60e223617a"
    print("[BOOT] Building indexes…")
    print("Knowledge:", ensure_knowledge_index())
    print("Account:", build_account_index(aid))
    print("[BOOT] Index build complete.")