# core/retrieval/knowledge_ingest.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext, Document
from llama_index.vector_stores.faiss import FaissVectorStore

PathLike = Union[str, Path]


def _embed_dim() -> int:
    """
    Return the embedding dimension you use elsewhere.
    If you use OpenAI text-embedding-3-large -> 3072; text-embedding-3-small -> 1536.
    Adjust if needed.
    """
    return 1536


def _collect_files(
    files: Optional[Sequence[PathLike]] = None,
    knowledge_dir: Optional[PathLike] = None,
    exts: Optional[Sequence[str]] = (".md", ".txt", ".pdf"),
) -> List[Path]:
    exts = tuple(exts or ())
    out: List[Path] = []

    if files:
        for f in files:
            p = Path(f)
            if p.is_file() and (not exts or p.suffix.lower() in exts):
                out.append(p)

    elif knowledge_dir:
        root = Path(knowledge_dir)
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and (not exts or p.suffix.lower() in exts):
                    out.append(p)

    return out


def ensure_knowledge_index(
    persist_dir: PathLike,
    *,
    files: Optional[Sequence[PathLike]] = None,
    knowledge_dir: Optional[PathLike] = None,
) -> dict:
    """
    Build (or rebuild) a FAISS knowledge index from either a `files` list or a `knowledge_dir`.

    Args:
        persist_dir: folder where the FAISS + docstore artifacts are written.
        files: specific files to include (preferred exact control).
        knowledge_dir: directory to crawl for *.md/*.txt/*.pdf (used if `files` is None).

    Returns:
        dict meta with count, dim, and persist_dir.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    picked = _collect_files(files=files, knowledge_dir=knowledge_dir)
    if not picked:
        # nothing to index; leave quietly
        return {"count": 0, "dim": _embed_dim(), "persist_dir": str(persist_path)}

    # Read documents
    # Use input_files to avoid directory readers creating nested metadata surprises.
    reader = SimpleDirectoryReader(input_files=[str(p) for p in picked])
    docs: List[Document] = reader.load_data()

    # Build FAISS-backed index
    dim = _embed_dim()
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage,
        show_progress=True,
    )

    # Persist artifacts (docstore.json, index files, etc.)
    index.storage_context.persist(persist_dir=str(persist_path))

    return {"count": len(docs), "dim": dim, "persist_dir": str(persist_path)}