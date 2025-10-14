from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import faiss

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore

# ---- Helpers ---------------------------------------------------------------

CUSTOMER_DATA_DIR  = Path("src/api/contextApp/data/customer_data")
INDEXES_STORE_DIR  = Path("src/api/contextApp/indexesstore")  # keep 'indexesstore'

def _embed_dim() -> int:
    # Match your embedding model dimension; default to OpenAI text-embedding-3-large (3072)
    v = os.getenv("EMBED_DIM")
    return int(v) if v and v.isdigit() else 3072

def _account_json_paths(account_id: str, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    base = (base_dir or CUSTOMER_DATA_DIR) / account_id
    return {
        "transactions":   base / "transactions.json",
        "payments":       base / "payments.json",
        "statements":     base / "statements.json",
        "account_summary": base / "account_summary.json",
    }

def _rows_from_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # If file happens to be ANSI, fall back without crashing
        obj = json.loads(path.read_text(errors="ignore"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # allow { "items": [...] } shapes
        for k in ("items", "transactions", "payments", "statements", "rows"):
            if isinstance(obj.get(k), list):
                return obj[k]
        return [obj]
    return []

# ---- Builders / Loaders ----------------------------------------------------

def build_account_index(
    account_id: str,
    base_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
) -> VectorStoreIndex:
    """
    Force (re)build an account FAISS index from the 4 JSON files and persist it.
    """
    pdir = persist_dir or (INDEXES_STORE_DIR / "accounts" / account_id / "llama")
    pdir.mkdir(parents=True, exist_ok=True)

    paths = _account_json_paths(account_id, base_dir=base_dir)
    docs: List[Document] = []

    # Gather rows, keep full JSON as text + convenient metadata
    for domain, jpath in paths.items():
        for r in _rows_from_json(jpath):
            docs.append(
                Document(
                    text=json.dumps(r, ensure_ascii=False),
                    metadata={
                        "account_id": account_id,
                        "domain": domain,
                        # shortcuts (may be None depending on row)
                        "merchantName": r.get("merchantName"),
                        "transactionType": r.get("transactionType"),
                        "displayTransactionType": r.get("displayTransactionType"),
                        "amount": r.get("amount"),
                        "period": r.get("period"),
                        "postedDateTime": r.get("postedDateTime"),
                        "transactionDateTime": r.get("transactionDateTime"),
                        "closingDateTime": r.get("closingDateTime"),
                        "paymentPostedDateTime": r.get("paymentPostedDateTime"),
                        "source": str(jpath),
                    },
                )
            )

    if not docs:
        # Persist empty store (still creates the folder structure) and return an empty index
        dim = _embed_dim()
        faiss_index = faiss.IndexFlatIP(dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(pdir))
        idx = VectorStoreIndex.from_documents([], storage_context=storage, show_progress=False)
        storage.persist(persist_dir=str(pdir))
        return idx

    dim = _embed_dim()
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(pdir))
    index = VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist(persist_dir=str(pdir))
    print(f"[INDEX] Built account index for {account_id} with {len(docs)} docs at {pdir}")
    return index


def ensure_account_index(
    account_id: str,
    base_dir: Optional[Path] = None,
    persist_root: Optional[Path] = None,
    force_rebuild: bool = False,
) -> VectorStoreIndex:
    """
    Idempotent loader:
      - if an index for `account_id` exists: load & return it
      - else: build it (via build_account_index) and return it
    """
    pdir = (persist_root or INDEXES_STORE_DIR) / "accounts" / account_id / "llama"
    pdir.mkdir(parents=True, exist_ok=True)

    # LlamaIndex writes index_store.json when persisted
    if (pdir / "index_store.json").exists() and not force_rebuild:
        vector_store = FaissVectorStore.from_persist_dir(str(pdir))
        storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(pdir))
        print(f"[INDEX] Loaded account index for {account_id} from {pdir}")
        return load_index_from_storage(storage)

    # Else build
    return build_account_index(account_id=account_id, base_dir=base_dir, persist_dir=pdir)

# ---- LangChain-compatible retriever ----------------------------------------

from langchain.schema import Document as LCDocument
from langchain.schema.retriever import BaseRetriever

class _LlamaIndexToLangchainRetriever(BaseRetriever):
    def __init__(self, li_retriever):
        super().__init__()
        self._li_retriever = li_retriever

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LCDocument]:
        nodes = self._li_retriever.retrieve(query)
        out: List[LCDocument] = []
        for n in nodes:
            text = getattr(n, "text", None) or getattr(n, "get_content", lambda: "")()
            meta = dict(getattr(n, "metadata", {}) or {})
            out.append(LCDocument(page_content=text, metadata=meta))
        return out

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[LCDocument]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def ensure_account_retriever(account_id: str, k: int = 6):
    """
    Load the persisted FAISS-backed LlamaIndex for this account and return
    a LangChain-compatible retriever wrapper.
    """
    base = Path(INDEX_STORE_DIR) / "accounts" / account_id / "llama"
    vector = FaissVectorStore.from_persist_dir(str(base))
    storage = StorageContext.from_defaults(vector_store=vector, persist_dir=str(base))
    index = load_index_from_storage(storage)

    # IMPORTANT: build a LlamaIndex RETRIEVER
    li_retriever = index.as_retriever(similarity_top_k=k)

    # Wrap for LC chains
    return _LlamaIndexToLangchainRetriever(li_retriever=li_retriever)return _LlamaIndexToLangchainRetriever(li_retriever)