# core/index/index_builder.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---- LlamaIndex (FAISS persist) ----
from llama_index.core import Document, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings as LISettings
from llama_index.vector_stores.faiss import FaissVectorStore

# ---- Your domain loaders (respect data.provider local/api) ----
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

import yaml


# =========================
# Config helpers
# =========================
def _read_cfg(cfg_path: str = "config/app.yaml") -> Dict[str, Any]:
    p = Path(cfg_path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _index_root(cfg: Dict[str, Any]) -> Path:
    return Path(((cfg.get("indexes") or {}).get("dir")) or "var/indexes")


def _should_rebuild(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("indexes") or {}).get("rebuild_on_startup", False))


# =========================
# Account discovery
# =========================
def _discover_accounts(cfg: Dict[str, Any]) -> List[str]:
    """
    For POC (local): treat each subfolder of data.accounts.root as an account id.
    For API mode: allow explicit list via config.data.accounts: [ "A1", "A2" ].
    """
    data_cfg = (cfg.get("data") or {})
    provider = (data_cfg.get("provider") or "local").lower()

    # API: prefer explicit allowlist
    if provider == "api":
        ids = data_cfg.get("accounts")
        if isinstance(ids, list) and ids:
            return [str(x) for x in ids]
        # If API has an endpoint to list accounts, you can add that later.
        print("[index_builder] data.provider=api but no data.accounts list; skipping account indexes.")
        return []

    # Local: scan local_root/<account_id>/
    root = Path((data_cfg.get("local_root") or "data/accounts"))
    if not root.exists():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir():
            # consider directory an account if it contains any of our jsons
            has_any = any((child / f).exists() for f in (
                "transactions.json", "payments.json", "statements.json", "account_summary.json"))
            if has_any:
                out.append(child.name)
    return sorted(out)


# =========================
# JSON → Documents
# =========================
def _doc_from_row(domain: str, acc_id: str, row: Dict[str, Any], path_hint: str) -> Document:
    """
    Flatten a record into a compact text summary with useful fields pronounced.
    Keep full JSON in metadata for grounding/audit.
    """
    # Pull a few common fields first (front-load relevance)
    pieces: List[str] = [f"domain={domain}", f"accountId={acc_id}"]
    # transactions
    if "merchantName" in row: pieces.append(f"merchant={row.get('merchantName')}")
    if "description" in row:  pieces.append(f"desc={row.get('description')}")
    if "amount" in row:       pieces.append(f"amount={row.get('amount')}")
    # payments
    if "paymentAmount" in row and "amount" not in row: pieces.append(f"amount={row.get('paymentAmount')}")
    # statements
    if "period" in row:       pieces.append(f"period={row.get('period')}")
    if "interestCharged" in row: pieces.append(f"interest={row.get('interestCharged')}")

    # any timestamp-like
    for k in ("postedDateTime","transactionDateTime","paymentPostedDateTime","paymentDateTime",
              "closingDateTime","openingDateTime","date"):
        if k in row and row[k]:
            pieces.append(f"{k}={row[k]}"); break

    # compact headline + full JSON tail
    head = " | ".join(pieces)
    tail = json.dumps(row, ensure_ascii=False)
    text = f"{head}\n{tail}"

    meta = {
        "source": path_hint,
        "domain": domain,
        "accountId": acc_id,
    }
    return Document(text=text, metadata=meta)


def _docs_for_account(cfg: Dict[str, Any], account_id: str) -> List[Document]:
    """
    Load all four JSONs for an account and convert to Documents.
    """
    docs: List[Document] = []
    # Each loader is already abstracted (local/api)
    txns = load_transactions(account_id, cfg) or []
    pays = load_payments(account_id, cfg) or []
    stmts = load_statements(account_id, cfg) or []
    acct  = load_account_summary(account_id, cfg) or {}

    for t in txns:
        if isinstance(t, dict):
            docs.append(_doc_from_row("transactions", account_id, t, f"transactions:{account_id}"))
    for p in pays:
        if isinstance(p, dict):
            docs.append(_doc_from_row("payments", account_id, p, f"payments:{account_id}"))
    for s in stmts:
        if isinstance(s, dict):
            docs.append(_doc_from_row("statements", account_id, s, f"statements:{account_id}"))

    if isinstance(acct, dict) and acct:
        # store the account summary as a single document with full JSON
        head = f"domain=accounts | accountId={account_id}"
        text = f"{head}\n{json.dumps(acct, ensure_ascii=False)}"
        docs.append(Document(text=text, metadata={"source": f"account_summary:{account_id}",
                                                  "domain": "accounts", "accountId": account_id}))
    return docs


# =========================
# Build / Load helpers
# =========================
def _persist_index_from_docs(docs: List[Document], out_dir: Path) -> None:
    """
    Idempotent: if index exists and rebuild not requested, we don't touch it.
    Else build a FAISS-backed VectorStoreIndex and persist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FaissVectorStore()  # dimension inferred from embedding model via LlamaIndex
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(out_dir))
    _ = VectorStoreIndex.from_documents(docs, storage_context=storage)
    storage.persist(persist_dir=str(out_dir))


def _index_exists(dir_path: Path) -> bool:
    # minimal heuristic; you can add more files if your persist format differs
    return (dir_path / "faiss.index").exists() or (dir_path / "index_store.json").exists()


# =========================
# Public: ensure account & knowledge indexes
# =========================
def ensure_account_index(cfg: Dict[str, Any], account_id: str, force_rebuild: bool = False) -> Dict[str, Any]:
    root = _index_root(cfg)
    out_dir = root / "accounts" / str(account_id)
    if _index_exists(out_dir) and not (force_rebuild or _should_rebuild(cfg)):
        return {"account_id": account_id, "persist_dir": out_dir.as_posix(), "status": "exists"}

    docs = _docs_for_account(cfg, account_id)
    if not docs:
        return {"account_id": account_id, "persist_dir": out_dir.as_posix(), "status": "skipped", "reason": "no_docs"}

    _persist_index_from_docs(docs, out_dir)
    return {"account_id": account_id, "persist_dir": out_dir.as_posix(), "status": "built", "count": len(docs)}


def ensure_all_account_indexes(cfg: Dict[str, Any], force_rebuild: bool = False) -> List[Dict[str, Any]]:
    ids = _discover_accounts(cfg)
    out: List[Dict[str, Any]] = []
    for acc in ids:
        try:
            out.append(ensure_account_index(cfg, acc, force_rebuild=force_rebuild))
        except Exception as e:
            out.append({"account_id": acc, "status": "error", "error": f"{type(e).__name__}: {e}"})
    return out


def ensure_knowledge_index(cfg: Dict[str, Any], force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Build a single knowledge index from handbook + policy/agreement docs.
    Defaults:
      - data/knowledge/ (e.g., handbook.md, .txt)
      - data/agreement/ (e.g., Apple-Card-Customer-Agreement.pdf)
    You can override via config.knowledge.sources: [ "dir1", "dir2", "fileX.pdf" ]
    """
    root = _index_root(cfg)
    out_dir = root / "knowledge"
    if _index_exists(out_dir) and not (force_rebuild or _should_rebuild(cfg)):
        return {"persist_dir": out_dir.as_posix(), "status": "exists"}

    # Collect sources
    kcfg = (cfg.get("knowledge") or {})
    custom = kcfg.get("sources")
    sources: List[Path] = []
    if isinstance(custom, list) and custom:
        sources = [Path(s) for s in custom]
    else:
        sources = [Path("data/knowledge"), Path("data/agreement")]

    # Read files (md, txt, pdf by default)
    files: List[Path] = []
    for s in sources:
        if s.is_dir():
            for p in s.rglob("*"):
                if p.suffix.lower() in {".md", ".txt", ".pdf"}:
                    files.append(p)
        elif s.is_file():
            files.append(s)

    if not files:
        return {"persist_dir": out_dir.as_posix(), "status": "skipped", "reason": "no_sources"}

    reader = SimpleDirectoryReader(input_files=[str(p) for p in files])
    docs = reader.load_data()
    if not docs:
        return {"persist_dir": out_dir.as_posix(), "status": "skipped", "reason": "empty_docs"}

    _persist_index_from_docs(docs, out_dir)
    return {"persist_dir": out_dir.as_posix(), "status": "built", "count": len(docs)}


# =========================
# One-call orchestrators
# =========================
def ensure_all_indexes(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Idempotent. Uses LlamaIndex Settings.llm/embed_model provided by your runtime at startup.
    """
    force = _should_rebuild(cfg)
    acc = ensure_all_account_indexes(cfg, force_rebuild=force)
    kn  = ensure_knowledge_index(cfg, force_rebuild=force)
    return {"accounts": acc, "knowledge": kn}


# Async-friendly wrapper if your runtime calls it with `await`.
async def ensure_indexes_startup(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return ensure_all_indexes(cfg)


# CLI (optional): `python -m core.index.index_builder`
if __name__ == "__main__":
    cfg = _read_cfg("config/app.yaml")
    out = ensure_all_indexes(cfg)
    print(json.dumps(out, indent=2))