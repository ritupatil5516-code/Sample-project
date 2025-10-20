# core/retrieval/index_builder.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ---- LlamaIndex / FAISS ----
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore

# Simple file reader that supports .md/.txt/.pdf (via pypdf)
from llama_index.core import SimpleDirectoryReader

# Local embedding as a sensible default (no OpenAI key required)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faiss is required. pip install faiss-cpu (or faiss-gpu) for your platform."
    ) from e


# ======================================================================================
# Defaults — change these to match your project structure
# ======================================================================================

# Where JSON lives
ACCOUNTS_DATA_DIR = Path("src/api/contextApp/customer_data")

# Knowledge files (md/pdf/txt)
KNOWLEDGE_DATA_DIR = Path("src/api/contextApp/data/knowledge")

# Where FAISS indices are persisted
INDEX_STORE_DIR = Path("src/api/contextApp/indexesstore")

# Subpaths for account / knowledge indices
ACCOUNTS_INDEX_DIR = INDEX_STORE_DIR / "accounts"
KNOWLEDGE_INDEX_DIR = INDEX_STORE_DIR / "knowledge" / "llama"


# ======================================================================================
# Utilities
# ======================================================================================

@dataclass
class BuildResult:
    count: int
    persist_dir: str
    dim: int


def _ensure_embed_model_if_missing() -> None:
    """
    Ensure LlamaIndex has an embedding model set.
    We default to a local HF model to avoid OpenAI dependency.
    """
    if Settings.embed_model is None:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


def _embedding_dim() -> int:
    """Probe the configured embedding model to determine vector dimension."""
    _ensure_embed_model_if_missing()
    model = Settings.embed_model
    for attr in ("get_text_embedding", "get_query_embedding", "embed", "embed_query"):
        fn = getattr(model, attr, None)
        if callable(fn):
            vec = fn("dimension probe")
            return len(vec)
    # Fallback: try attribute
    dim = getattr(model, "dimension", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    raise RuntimeError("Could not determine embedding dimension from the embed model.")


def _mk_storage_with_faiss(persist_dir: Path) -> Tuple[StorageContext, FaissVectorStore]:
    """Create a FAISS vector store + storage context for a given persist directory."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    dim = _embedding_dim()
    faiss_index = faiss.IndexFlatIP(dim)  # cosine w/ normalized vectors; or L2 via IndexFlatL2
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))
    return storage, vector_store


def _iter_files(root: Union[str, Path], exts: Sequence[str]) -> List[Path]:
    p = Path(root)
    if not p.exists():
        return []
    out: List[Path] = []
    for ext in exts:
        out.extend(p.rglob(f"*{ext}"))
    return out


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _yyyy_mm_from_iso(dt: Optional[str]) -> Optional[str]:
    if not dt or not isinstance(dt, str):
        return None
    # Expecting 2024-09-01T... or 2024-09-01
    try:
        return dt[:7]
    except Exception:
        return None


# ======================================================================================
# Knowledge indexing
# ======================================================================================

def ensure_knowledge_index(
    knowledge_dir: Union[str, Path] = KNOWLEDGE_DATA_DIR,
    persist_dir: Union[str, Path] = KNOWLEDGE_INDEX_DIR,
    files: Optional[Union[Path, str, Iterable[Union[Path, str]]]] = None,
) -> BuildResult:
    """
    Build (or rebuild) the knowledge FAISS index.

    - knowledge_dir: folder that contains .md/.pdf/.txt
    - files: optional list/iterable of explicit file paths; also accepts a single Path/str
             (we normalize single values → [value] to avoid `WindowsPath not iterable`)
    - persist_dir: location for FAISS + index/docstore json

    Returns: BuildResult(count=#docs, persist_dir, dim)
    """
    persist_dir = Path(persist_dir)
    knowledge_dir = Path(knowledge_dir)

    # Resolve file set
    if files is None:
        file_paths = _iter_files(knowledge_dir, exts=[".md", ".pdf", ".txt"])
    else:
        # normalize to list of Paths
        if isinstance(files, (str, Path)):
            file_paths = [Path(files)]
        else:
            file_paths = [Path(x) for x in files]

    if not file_paths:
        return BuildResult(count=0, persist_dir=str(persist_dir), dim=_embedding_dim())

    # Load documents
    reader = SimpleDirectoryReader(input_files=[str(p) for p in file_paths])
    docs = reader.load_data()

    # Build FAISS vector store and persist
    storage, _ = _mk_storage_with_faiss(persist_dir)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist()

    return BuildResult(count=len(docs), persist_dir=str(persist_dir), dim=_embedding_dim())


# ======================================================================================
# Account indexing (transactions/payments/statements/account_summary)
# ======================================================================================

def _account_paths(account_id: str, base_dir: Union[str, Path]) -> Dict[str, Path]:
    """Return expected JSON file paths for an account."""
    root = Path(base_dir) / account_id
    return {
        "transactions": root / "transactions.json",
        "payments": root / "payments.json",
        "statements": root / "statements.json",
        "account_summary": root / "account_summary.json",
    }


def _rows_from_json(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        # either dict with "items" or a single object; normalize to one row
        if "items" in obj and isinstance(obj["items"], list):
            return list(obj["items"])
        return [obj]
    return []


def build_account_index(
    account_id: str,
    base_dir: Union[str, Path] = ACCOUNTS_DATA_DIR,
    persist_dir: Optional[Union[str, Path]] = None,
) -> BuildResult:
    """
    Build an account-specific FAISS index from JSON domain files.
    Documents are the raw JSON rows per domain with helpful metadata.
    """
    if persist_dir is None:
        persist_dir = ACCOUNTS_INDEX_DIR / account_id / "llama"
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    paths = _account_paths(account_id, base_dir)
    tx_p = paths["transactions"]
    pay_p = paths["payments"]
    stm_p = paths["statements"]
    sum_p = paths["account_summary"]

    docs: List[Document] = []

    def add_docs(rows: List[Dict[str, Any]], domain: str) -> None:
        for r in rows:
            # enrich metadata for common filters
            md: Dict[str, Any] = {
                "account_id": account_id,
                "domain": domain,
                "merchantName": r.get("merchantName"),
                "transactionType": r.get("transactionType") or r.get("displayTransactionType"),
                "amount": r.get("amount"),
                "period": (
                    r.get("period")
                    or _yyyy_mm_from_iso(r.get("postedDateTime"))
                    or _yyyy_mm_from_iso(r.get("transactionDateTime"))
                    or _yyyy_mm_from_iso(r.get("closingDateTime"))
                ),
                "postedDateTime": r.get("postedDateTime"),
                "transactionDateTime": r.get("transactionDateTime"),
                "closingDateTime": r.get("closingDateTime"),
            }
            docs.append(
                Document(text=json.dumps(r, ensure_ascii=False), metadata=md)
            )

    # transactions
    if tx_p.exists():
        add_docs(_rows_from_json(_read_json(tx_p)), "transactions")

    # payments
    if pay_p.exists():
        add_docs(_rows_from_json(_read_json(pay_p)), "payments")

    # statements
    if stm_p.exists():
        add_docs(_rows_from_json(_read_json(stm_p)), "statements")

    # account_summary (usually just one row)
    if sum_p.exists():
        add_docs(_rows_from_json(_read_json(sum_p)), "account_summary")

    if not docs:
        return BuildResult(count=0, persist_dir=str(persist_dir), dim=_embedding_dim())

    storage, _ = _mk_storage_with_faiss(persist_dir)
    VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist()

    return BuildResult(count=len(docs), persist_dir=str(persist_dir), dim=_embedding_dim())


# ======================================================================================
# Convenience: build all
# ======================================================================================

def build_all(
    account_id: Optional[str] = None,
    accounts_base_dir: Union[str, Path] = ACCOUNTS_DATA_DIR,
    knowledge_dir: Union[str, Path] = KNOWLEDGE_DATA_DIR,
) -> Dict[str, Any]:
    """
    Build knowledge index and (optionally) the account index.
    Returns a small summary dict.
    """
    # ensure an embedder exists for both builds
    _ensure_embed_model_if_missing()

    out: Dict[str, Any] = {"index_root": str(INDEX_STORE_DIR)}

    # Knowledge
    try:
        kr = ensure_knowledge_index(knowledge_dir=knowledge_dir, persist_dir=KNOWLEDGE_INDEX_DIR)
        out["knowledge"] = {"count": kr.count, "persist_dir": kr.persist_dir, "dim": kr.dim}
    except Exception as e:
        out["knowledge"] = {"error": f"{type(e).__name__}: {e}"}

    # Account (if requested)
    if account_id:
        try:
            ar = build_account_index(account_id=account_id, base_dir=accounts_base_dir)
            out["account"] = {"count": ar.count, "persist_dir": ar.persist_dir, "dim": ar.dim}
        except Exception as e:
            out["account"] = {"error": f"{type(e).__name__}: {e}"}

    return out


# ======================================================================================
# CLI
# ======================================================================================

if __name__ == "__main__":
    """
    Example usage:
      python -m core.retrieval.index_builder                      # knowledge only
      python -m core.retrieval.index_builder 3b1ba69f-...-3617a   # also build account index
    """
    import sys

    # Make sure we have some embedding model even when run standalone.
    _ensure_embed_model_if_missing()

    acct_id = sys.argv[1] if len(sys.argv) > 1 else None

    print("[BOOT] Building all indexes…")
    res = build_all(
        account_id=acct_id,
        accounts_base_dir=ACCOUNTS_DATA_DIR,
        knowledge_dir=KNOWLEDGE_DATA_DIR,
    )
    print("[BOOT] Index build complete.")
    print(json.dumps(res, indent=2))