# core/retrieval/knowledge_ingest.py
from __future__ import annotations
from pathlib import Path
from typing import List
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _read_cfg():
    import yaml, json
    p = Path("config/app.yaml")
    return (yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {})

def _load_source_dir(root:Path) -> List[str]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".md",".txt",".json"):
            files.append(p.as_posix())
    return files

def ensure_knowledge_retriever(k:int=6):
    cfg = _read_cfg()
    idx_dir = Path(((cfg.get("indexes") or {}).get("dir")) or "api/contextApp/indexstore")
    store = idx_dir/"knowledge"
    store.mkdir(parents=True, exist_ok=True)
    emb = OpenAIEmbeddings(
        model=(cfg.get("embeddings") or {}).get("openai_model","text-embedding-3-large"),
        api_key=os.getenv((cfg.get("embeddings") or {}).get("openai_api_key_env","OPENAI_API_KEY"),""),
        base_url=(cfg.get("embeddings") or {}).get("openai_base_url","https://api.openai.com/v1")
    )
    try:
        vs = FAISS.load_local(store.as_posix(), emb, allow_dangerous_deserialization=True)
        return vs.as_retriever(search_kwargs={"k":k})
    except Exception:
        pass

    # build from /data/knowledge
    root = Path("src/api/contextApp/data/knowledge")
    root.mkdir(parents=True, exist_ok=True)
    files = _load_source_dir(root)
    if not files:
        # create a tiny placeholder so queries don't 500
        open((root/"README.txt"),"w",encoding="utf-8").write("Knowledge base placeholder.")
        files = [ (root/"README.txt").as_posix() ]

    docs = []
    for f in files:
        docs += TextLoader(f, encoding="utf-8").load()

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(store.as_posix())
    return vs.as_retriever(search_kwargs={"k":k})