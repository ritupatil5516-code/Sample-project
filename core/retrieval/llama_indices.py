# core/retrieval/llama_indices.py
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding  # uses OPENAI_API_KEY
from llama_index.core import Settings as LISettings

LISettings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

INDEX_ROOT = Path("var/indexes_llama")

def _json_rows_to_docs(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Document]:
    # stringify a sensible view per row; keep rich metadata
    docs = []
    for r in rows:
        text = "\n".join(f"{k}: {v}" for k, v in r.items())
        docs.append(Document(text=text, metadata={**meta, **r}))
    return docs

def ensure_account_index(account_id: str,
                         txns: List[Dict[str, Any]],
                         pays: List[Dict[str, Any]],
                         stmts: List[Dict[str, Any]],
                         acct: Dict[str, Any]) -> VectorStoreIndex:
    outdir = INDEX_ROOT / "accounts" / account_id
    if outdir.exists():
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(outdir)))
    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=120)
    pipe = IngestionPipeline(transformations=[splitter])
    docs = []
    docs += _json_rows_to_docs(txns, {"domain":"transactions", "accountId":account_id})
    docs += _json_rows_to_docs(pays,  {"domain":"payments",     "accountId":account_id})
    docs += _json_rows_to_docs(stmts, {"domain":"statements",   "accountId":account_id})
    docs += _json_rows_to_docs([acct],{"domain":"account_summary","accountId":account_id})
    nodes = pipe.run(documents=docs)
    index = VectorStoreIndex.from_documents(nodes)  # defaults to in-memory + persisted
    index.storage_context.persist(persist_dir=str(outdir))
    return index

def ensure_knowledge_index(paths: List[str]) -> VectorStoreIndex:
    outdir = INDEX_ROOT / "knowledge"
    if outdir.exists():
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(outdir)))
    # load files (handbook.md, Apple-Card-Customer-Agreement.pdf) using LI readers
    from llama_index.readers.file import FlatReader
    loader = FlatReader()
    docs = []
    for p in paths:
        for d in loader.load_data(Path(p)):
            d.metadata.update({"domain":"knowledge", "source": p})
            docs.append(d)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(outdir))
    return index

def account_retriever(account_index, k=5):
    return account_index.as_retriever(similarity_top_k=k)

def knowledge_retriever(knowledge_index, k=5):
    return knowledge_index.as_retriever(similarity_top_k=k)