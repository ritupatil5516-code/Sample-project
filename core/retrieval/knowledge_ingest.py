# core/retrieval/knowledge_ingest.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any, List

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


def _embed_dim() -> int:
    """
    Try to read the embedding dimension from the configured embed model.
    Fall back to 1536 if not available.
    """
    return int(getattr(Settings.embed_model, "embed_dim", 1536))


def ensure_knowledge_index(
    knowledge_dir: str | Path,
    persist_dir: str | Path,
    files: Sequence[str | Path] | None = None,
) -> Dict[str, Any]:
    """
    Build (or reuse) a FAISS-backed index over knowledge files (handbook, agreement, FAQs, etc.)

    Parameters
    ----------
    knowledge_dir : str | Path
        Directory that contains knowledge files (used if `files` is None).
    persist_dir : str | Path
        Where to persist the FAISS + storage artifacts.
    files : Sequence[str | Path] | None
        Optional explicit list of files to index. Must be an iterable.
        Example: [Path("data/knowledge/handbook.md"), Path("data/agreement/Apple-Card.pdf")]

    Returns
    -------
    Dict[str, Any]
        Meta about the built index: count, persist_dir, dim.
    """
    knowledge_dir = Path(knowledge_dir).resolve()
    persist_dir = Path(persist_dir).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load documents
    if files:
        # IMPORTANT: must be an iterable; pass absolute file paths
        input_files = [str(Path(f).resolve()) for f in files]
        reader = SimpleDirectoryReader(input_files=input_files)
    else:
        reader = SimpleDirectoryReader(input_dir=str(knowledge_dir))

    docs: List[Document] = reader.load_data()

    # ---- Build FAISS store
    dim = _embed_dim()
    faiss_index = faiss.IndexFlatIP(dim)  # use cosine-sim via inner product; change to L2 if you prefer
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))

    # ---- Create index & persist
    index = VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist(persist_dir=str(persist_dir))

    return {"count": len(docs), "persist_dir": str(persist_dir), "dim": dim}