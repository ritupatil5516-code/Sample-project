# src/api/contextApp/core/retrieval/knowledge_ingest.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import os
import faiss

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore

# ---------------- LangChain adapter (LlamaIndex retriever -> LC retriever) ---
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
            meta = dict(getattr(n, "metadata", {}) or {})
            text = getattr(n, "text", None) or getattr(n, "get_content", lambda: "")()
            out.append(LCDocument(page_content=text, metadata=meta))
        return out

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[LCDocument]:
        return self._get_relevant_documents(query, run_manager=run_manager)


# ------------------------------ Paths / constants -----------------------------
# Adjust if your repo layout differs
KNOWLEDGE_DATA_DIR = Path("src/api/contextApp/data/knowledge")
INDEXES_STORE_DIR  = Path("src/api/contextApp/indexesstore")  # keep 'indexesstore'

ALLOWED_EXTS = {".md", ".txt", ".pdf"}

def _embed_dim() -> int:
    """Match your embedding model dimension."""
    env = os.getenv("EMBED_DIM")
    if env and env.isdigit():
        return int(env)
    # Default to OpenAI text-embedding-3-large; change if you use Qwen (4096) or 3-small (1536)
    return 3072


# ------------------------------ File loaders ---------------------------------
def _read_pdf_text(path: Path) -> str:
    """Lightweight PDF text extractor; uses pypdf if present."""
    try:
        from pypdf import PdfReader  # pip install pypdf
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception:
        # Fall back: no PDF backend available
        print(f"[WARN] Could not import pypdf to read {path.name}; skipping PDF text.")
        return ""


def _load_knowledge_docs(knowledge_dir: Path) -> List[Document]:
    """
    Load Markdown / TXT / PDF into LlamaIndex Documents with minimal metadata.
    Uses SimpleDirectoryReader for MD/TXT; PDFs go through pypdf (if available).
    """
    docs: List[Document] = []

    # 1) Try to use SimpleDirectoryReader for non-PDFs
    if knowledge_dir.exists():
        try:
            sdr = SimpleDirectoryReader(
                input_dir=str(knowledge_dir),
                required_exts=list(ALLOWED_EXTS - {".pdf"}),
                recursive=False,
            )
            docs.extend(sdr.load_data())
        except Exception as e:
            print(f"[WARN] SimpleDirectoryReader non-PDF failed: {e}")

    # 2) PDFs (manually) so we don't depend on unstructured stack
    for p in knowledge_dir.glob("*.pdf"):
        txt = _read_pdf_text(p)
        if txt.strip():
            docs.append(
                Document(
                    text=txt,
                    metadata={"source": str(p), "filename": p.name, "ext": ".pdf"},
                )
            )

    # 3) If user provided explicit file list under knowledge_dir (optional)
    #    You can add additional single-file loads here if needed.

    # Final sanity
    # Some loaders (older LI versions) might return Node-like objs; adapt them
    real_docs: List[Document] = []
    for d in docs:
        if isinstance(d, Document):
            real_docs.append(d)
        else:
            # Defensive: wrap as Document
            text = getattr(d, "text", None) or getattr(d, "get_content", lambda: "")()
            meta = dict(getattr(d, "metadata", {}) or {})
            real_docs.append(Document(text=text, metadata=meta))

    return real_docs


# -------------------------- Build / Load knowledge index ----------------------
def ensure_knowledge_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """
    Build the FAISS knowledge index if missing (or force rebuild), else load it.
    Returns a LlamaIndex VectorStoreIndex.
    """
    persist_dir = INDEXES_STORE_DIR / "knowledge" / "llama"
    persist_dir.mkdir(parents=True, exist_ok=True)

    index_store_path = persist_dir / "index_store.json"
    if index_store_path.exists() and not force_rebuild:
        # Load existing
        vector_store = FaissVectorStore.from_persist_dir(str(persist_dir))
        storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))
        print(f"[INDEX] Loaded knowledge index from {persist_dir}")
        return load_index_from_storage(storage)

    # Build
    docs = _load_knowledge_docs(KNOWLEDGE_DATA_DIR)
    dim = _embed_dim()
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))
    index = VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist(persist_dir=str(persist_dir))
    print(f"[INDEX] Built knowledge index with {len(docs)} docs at {persist_dir}")
    return index


def load_knowledge_index(persist_dir: str | Path) -> VectorStoreIndex:
    """Explicit loader (if you want to mount a custom path)."""
    p = Path(persist_dir)
    vector_store = FaissVectorStore.from_persist_dir(str(p))
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(p))
    return load_index_from_storage(storage)


# ----------------------------- Public: retriever ------------------------------
def ensure_knowledge_retriever(k: int = 6, as_langchain: bool = True):
    """
    Returns a retriever over the knowledge corpus (handbook.md, policy PDF, etc.).
    - as_langchain=True (default) returns a LangChain BaseRetriever.
    - as_langchain=False returns a LlamaIndex retriever.
    """
    index = ensure_knowledge_index(force_rebuild=False)
    li_retriever = index.as_retriever(similarity_top_k=k)
    if not as_langchain:
        return li_retriever
    return _LlamaIndexToLangchainRetriever(li_retriever)