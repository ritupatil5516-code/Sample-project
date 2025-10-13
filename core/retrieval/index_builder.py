# core/retrieval/index_builder.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any

from llama_index.core import Document, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore

# ---------- Defaults (override via config/app.yaml if you want) ----------
BASE_DIR          = Path("src/api/contextApp")
CUSTOMER_DATA_DIR = BASE_DIR / "customer_data"
INDEXES_DIR       = BASE_DIR / "indexesstore"
KNOWLEDGE_INDEX   = INDEXES_DIR / "_knowledge"

DEFAULT_KNOWLEDGE_FILES = [
    "data/knowledge/handbook.md",
    "data/agreement/Apple-Card-Customer-Agreement.pdf",
]

# ---------- tiny config reader (non-fatal if missing) ----------
def _read_cfg():
    p = Path("config/app.yaml")
    if not p.exists(): return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _json_file_to_documents(path: Path, domain: str, account_id: str) -> List[Document]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # fall back: treat raw text
        return [Document(text=path.read_text(encoding="utf-8"),
                         metadata={"domain": domain, "account_id": account_id, "source": path.as_posix()})]

    docs: List[Document] = []
    if isinstance(data, list):
        it = data
    elif isinstance(data, dict):
        it = data.get("items") if isinstance(data.get("items"), list) else [data]
    else:
        it = [data]

    for i, item in enumerate(it):
        docs.append(
            Document(
                text=json.dumps(item, ensure_ascii=False),
                metadata={
                    "domain": domain,
                    "account_id": account_id,
                    "source": f"{path.name}#{i}"
                },
            )
        )
    return docs

def _build_faiss_index(docs: List[Document], persist_dir: Path, dim: int = 1536):
    persist_dir.mkdir(parents=True, exist_ok=True)
    # LlamaIndex + FAISS
    vs = FaissVectorStore(dim=dim)
    sc = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex.from_documents(docs, storage_context=sc, show_progress=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return persist_dir

# ---------- public: build account & knowledge indexes ----------

def build_account_index(account_id: str) -> Path:
    base = CUSTOMER_DATA_DIR / account_id
    if not base.exists():
        raise FileNotFoundError(f"Customer data not found: {base}")

    docs: List[Document] = []
    docs += _json_file_to_documents(base / "transactions.json",    "transactions",    account_id)
    docs += _json_file_to_documents(base / "payments.json",        "payments",        account_id)
    docs += _json_file_to_documents(base / "statements.json",      "statements",      account_id)
    docs += _json_file_to_documents(base / "account_summary.json", "account_summary", account_id)

    # allow per-account extras (optional)
    extras = [base / "policy.pdf", base / "handbook.pdf"]
    existing = [p for p in extras if p.exists()]
    if existing:
        docs += SimpleDirectoryReader(input_files=[p.as_posix() for p in existing]).load_data()

    if not docs:
        raise ValueError(f"No source documents found for {account_id}")

    return _build_faiss_index(docs, INDEXES_DIR / account_id)

def build_knowledge_index() -> Path:
    cfg = _read_cfg()
    paths = (cfg.get("knowledge") or {}).get("paths") or DEFAULT_KNOWLEDGE_FILES
    files = [Path(p) for p in paths if Path(p).exists()]
    if not files:
        # Non-fatal: create an empty but valid index so loads don’t crash later.
        docs = [Document(text="(empty knowledge)")]
    else:
        docs = SimpleDirectoryReader(input_files=[f.as_posix() for f in files]).load_data()
    return _build_faiss_index(docs, KNOWLEDGE_INDEX)

def build_all_indexes() -> None:
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    # knowledge (once)
    try:
        print("[INDEX] Building knowledge index…")
        build_knowledge_index()
        print(f"[INDEX] Knowledge index ready at {KNOWLEDGE_INDEX}")
    except Exception as e:
        print(f"[WARN] Knowledge index build failed: {e}")

    # accounts
    if not CUSTOMER_DATA_DIR.exists():
        print(f"[WARN] No customer_data dir at {CUSTOMER_DATA_DIR}")
        return

    for aid in sorted(p.name for p in CUSTOMER_DATA_DIR.iterdir() if p.is_dir()):
        try:
            print(f"[INDEX] Building account index for {aid}…")
            out = build_account_index(aid)
            print(f"[INDEX] Account index ready at {out}")
        except Exception as e:
            print(f"[WARN] Failed to build index for {aid}: {e}")