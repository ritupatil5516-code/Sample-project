# core/retrieval/json_ingest.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json, os

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def _read_cfg():
    import yaml
    p = Path("config/app.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}

def _account_dir(aid:str) -> Path:
    return Path("src/api/contextApp/data/customer_data")/aid

def _to_docs(records: List[Dict[str,Any]], *, source:str, doc_type:str) -> List[Document]:
    docs: List[Document] = []
    for i, r in enumerate(records):
        # flatten into readable text; keep raw in metadata
        text_parts = []
        for k,v in r.items():
            if isinstance(v,(dict,list)): continue
            text_parts.append(f"{k}: {v}")
        content = "\n".join(text_parts) or json.dumps(r, ensure_ascii=False)
        docs.append(Document(page_content=content, metadata={"source":source,"type":doc_type,"raw":r}))
    return docs

def _split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return splitter.split_documents(docs)

def _load_json_list(p: Path) -> List[Dict[str,Any]]:
    if not p.exists(): return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data,list) else [data]

def build_account_vectorstore(account_id:str, index_dir:Path) -> FAISS:
    acc = _account_dir(account_id)
    txns = _load_json_list(acc/"transactions.json")
    pays = _load_json_list(acc/"payments.json")
    stmts= _load_json_list(acc/"statements.json")
    acct = [json.loads((acc/"account_summary.json").read_text(encoding="utf-8"))] if (acc/"account_summary.json").exists() else []

    docs = []
    docs += _to_docs(txns,  source=f"{account_id}/transactions.json", doc_type="transactions")
    docs += _to_docs(pays,  source=f"{account_id}/payments.json",     doc_type="payments")
    docs += _to_docs(stmts, source=f"{account_id}/statements.json",   doc_type="statements")
    docs += _to_docs(acct,  source=f"{account_id}/account_summary.json", doc_type="account_summary")

    chunks = _split(docs)
    cfg = _read_cfg()
    emb = OpenAIEmbeddings(
        model=(cfg.get("embeddings") or {}).get("openai_model","text-embedding-3-large"),
        api_key=os.getenv((cfg.get("embeddings") or {}).get("openai_api_key_env","OPENAI_API_KEY"),""),
        base_url=(cfg.get("embeddings") or {}).get("openai_base_url","https://api.openai.com/v1")
    )
    vs = FAISS.from_documents(chunks, emb)
    (index_dir/f"accounts/{account_id}").mkdir(parents=True, exist_ok=True)
    vs.save_local((index_dir/f"accounts/{account_id}").as_posix())
    return vs

def load_account_vectorstore(account_id:str, index_dir:Path) -> FAISS:
    from langchain_community.vectorstores import FAISS
    path = index_dir/f"accounts/{account_id}"
    return FAISS.load_local(path.as_posix(), OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def ensure_account_retriever(account_id:str, k:int=6):
    cfg = _read_cfg()
    idx_dir = Path(((cfg.get("indexes") or {}).get("dir")) or "api/contextApp/indexstore")
    try:
        vs = load_account_vectorstore(account_id, idx_dir)
    except Exception:
        vs = build_account_vectorstore(account_id, idx_dir)
    return vs.as_retriever(search_kwargs={"k": k})