# core/retrieval/index_builder.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ---- LlamaIndex / FAISS ----
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import SimpleDirectoryReader

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faiss is required. pip install faiss-cpu (or faiss-gpu) for your platform."
    ) from e


# ============================== Paths / Layout =================================

ACCOUNTS_DATA_DIR  = Path("src/api/contextApp/customer_data")
KNOWLEDGE_DATA_DIR = Path("src/api/contextApp/data/knowledge")

INDEX_STORE_DIR     = Path("src/api/contextApp/indexesstore")
ACCOUNTS_INDEX_DIR  = INDEX_STORE_DIR / "accounts" / "{account_id}" / "llama"
KNOWLEDGE_INDEX_DIR = INDEX_STORE_DIR / "knowledge" / "llama"


# ============================== Small utilities ================================

@dataclass
class BuildResult:
    count: int
    persist_dir: str
    dim: int

def _ensure_embed_model() -> None:
    if Settings.embed_model is None:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

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

def _mk_storage_with_faiss(persist_dir: Path) -> Tuple[StorageContext, FaissVectorStore]:
    persist_dir.mkdir(parents=True, exist_ok=True)
    _ensure_embed_model()
    # probe dim
    vec = Settings.embed_model.get_text_embedding("probe")
    dim = len(vec)
    index = faiss.IndexFlatIP(dim)  # cosine (with normalized vectors)
    vector_store = FaissVectorStore(faiss_index=index)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))
    return storage, vector_store

def _first(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v:
            return v
    return None

def _yyyy_mm_from_iso(dt: Optional[str]) -> Optional[str]:
    return dt[:7] if isinstance(dt, str) and len(dt) >= 7 else None

def _amount_sign(row: Dict[str, Any]) -> str:
    """
    Normalize sign for finance rows:
      - debitCreditIndicator '1' => DEBIT (money leaving you)
      - debitCreditIndicator '-1' => CREDIT (refund/credit)
      - fallback by amount value
    """
    dci = str(row.get("debitCreditIndicator", "")).strip()
    if dci == "-1":
        return "CREDIT"
    if dci == "1":
        return "DEBIT"
    try:
        amt = float(row.get("amount", 0))
        return "DEBIT" if amt > 0 else "CREDIT"
    except Exception:
        return "UNKNOWN"

def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}

def _fmt_money(v: Any) -> str:
    try:
        x = float(v)
        sign = "-" if x < 0 else ""
        x = abs(x)
        return f"{sign}${x:,.2f}"
    except Exception:
        return str(v)


# ============================== Knowledge index ================================

def ensure_knowledge_index(
    knowledge_dir: Union[str, Path] = KNOWLEDGE_DATA_DIR,
    persist_dir: Union[str, Path] = KNOWLEDGE_INDEX_DIR,
    files: Optional[Union[Path, str, Iterable[Union[Path, str]]]] = None,
) -> BuildResult:
    _ensure_embed_model()
    persist_dir = Path(persist_dir)

    # Resolve file set
    if files is None:
        file_paths = _iter_files(knowledge_dir, [".md", ".pdf", ".txt"])
    else:
        if isinstance(files, (str, Path)):
            file_paths = [Path(files)]
        else:
            file_paths = [Path(x) for x in files]

    if not file_paths:
        # still return dim so caller knows embed model worked
        dim = len(Settings.embed_model.get_text_embedding("probe"))
        return BuildResult(count=0, persist_dir=str(persist_dir), dim=dim)

    reader = SimpleDirectoryReader(input_files=[str(p) for p in file_paths])
    docs = reader.load_data()

    storage, _ = _mk_storage_with_faiss(persist_dir)
    VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist()

    dim = len(Settings.embed_model.get_text_embedding("probe"))
    return BuildResult(count=len(docs), persist_dir=str(persist_dir), dim=dim)


# ============================== Account index ==================================

def _account_paths(account_id: str, base_dir: Union[str, Path]) -> Dict[str, Path]:
    root = Path(base_dir) / account_id
    return {
        "transactions": root / "transactions.json",
        "payments": root / "payments.json",
        "statements": root / "statements.json",
        "account_summary": root / "account_summary.json",
    }

def _rows(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        return obj.get("items", []) if isinstance(obj.get("items"), list) else [obj]
    return []

# ---------- Renderers (text for embedding) ----------

def _render_transaction(r: Dict[str, Any]) -> str:
    ts = _first(r.get("postedDateTime"), r.get("transactionDateTime"))
    amt = _fmt_money(r.get("amount"))
    m   = r.get("merchantName") or r.get("description") or ""
    cat = r.get("category") or r.get("merchantCategoryName") or ""
    disp= r.get("displayTransactionType") or r.get("transactionType") or ""
    stat= r.get("transactionStatus") or ""
    note= r.get("memo") or r.get("note") or ""
    sign= _amount_sign(r)
    parts = [
        "TRANSACTION",
        f"date={ts}" if ts else "",
        f"merchant={m}" if m else "",
        f"category={cat}" if cat else "",
        f"type={disp}",
        f"status={stat}" if stat else "",
        f"amount={amt}",
        f"sign={sign}",
    ]
    # mark interest explicitly
    if "interest" in str(disp).lower() or "interest" in str(m).lower():
        parts.append("tag=interest")
    if "refund" in str(disp).lower() or sign == "CREDIT":
        parts.append("tag=refund_or_credit")
    if note:
        parts.append(f"notes={note}")
    return " | ".join([p for p in parts if p])

def _render_payment(r: Dict[str, Any]) -> str:
    ts = _first(r.get("paymentPostedDateTime"), r.get("paymentDateTime"))
    amt = _fmt_money(r.get("amount"))
    src = (r.get("fundingSource") or {}).get("accountType") or r.get("fundingType") or ""
    stat= r.get("state") or r.get("status") or ""
    parts = [
        "PAYMENT",
        f"date={ts}" if ts else "",
        f"amount={amt}",
        f"status={stat}" if stat else "",
        f"source={src}" if src else "",
    ]
    return " | ".join([p for p in parts if p])

def _render_statement(r: Dict[str, Any]) -> str:
    close = r.get("closingDateTime")
    due   = _first(r.get("dueDateTime"), r.get("dueDate"))
    int_ch= r.get("interestCharged", 0.0)
    int_nt= r.get("totalNonTrailingInterest", r.get("nonTrailingInterest", 0.0))
    int_tr= r.get("totalTrailingInterest", r.get("trailingInterest", 0.0))
    purch = r.get("purchases", r.get("totalPurchased", None))
    pays  = r.get("totalPayments", r.get("paymentsAndCredits", None))
    parts = [
        "STATEMENT",
        f"closing={close}" if close else "",
        f"due={due}" if due else "",
        f"interestCharged={_fmt_money(int_ch)}",
        f"nonTrailing={_fmt_money(int_nt)}",
        f"trailing={_fmt_money(int_tr)}",
    ]
    if purch is not None: parts.append(f"purchases={_fmt_money(purch)}")
    if pays  is not None: parts.append(f"payments={_fmt_money(pays)}")
    if _bool(r.get("isTrailingInterestApplied")):
        parts.append("tag=trailing_interest_applied")
    return " | ".join([p for p in parts if p])

def _render_account_summary(r: Dict[str, Any]) -> str:
    bal = r.get("currentBalance", r.get("currentAdjustedBalance"))
    avail = r.get("availableCredit")
    cl   = r.get("creditLimit")
    status = r.get("accountStatus") or ""
    due_date = _first(r.get("paymentDueDateTime"), r.get("paymentDueDate"))
    parts = [
        "ACCOUNT_SUMMARY",
        f"status={status}" if status else "",
        f"currentBalance={_fmt_money(bal)}" if bal is not None else "",
        f"availableCredit={_fmt_money(avail)}" if avail is not None else "",
        f"creditLimit={_fmt_money(cl)}" if cl is not None else "",
        f"nextPaymentDue={due_date}" if due_date else "",
    ]
    return " | ".join([p for p in parts if p])

# ---------- Metadata builder ----------

def _base_meta(account_id: str, domain: str, r: Dict[str, Any]) -> Dict[str, Any]:
    ts = _first(
        r.get("postedDateTime"),
        r.get("transactionDateTime"),
        r.get("paymentPostedDateTime"),
        r.get("paymentDateTime"),
        r.get("closingDateTime"),
        r.get("date"),
    )
    ym = _yyyy_mm_from_iso(ts) or r.get("period")
    meta: Dict[str, Any] = {
        "account_id": account_id,
        "domain": domain,
        "ts": ts,
        "ym": ym,
        "amount": r.get("amount"),
        "amount_sign": _amount_sign(r),
        "merchantName": r.get("merchantName"),
        "category": r.get("category") or r.get("merchantCategoryName"),
        "transactionType": r.get("transactionType") or r.get("displayTransactionType"),
        "transactionStatus": r.get("transactionStatus") or r.get("status"),
        "closingDateTime": r.get("closingDateTime"),
        "paymentPostedDateTime": r.get("paymentPostedDateTime"),
        "postedDateTime": r.get("postedDateTime"),
        "transactionDateTime": r.get("transactionDateTime"),
    }
    # interest flags
    disp = str(r.get("displayTransactionType") or "").lower()
    m    = str(r.get("merchantName") or "").lower()
    meta["is_interest_charge_tx"]  = ("interest" in disp and "credit" not in disp) or ("interest" in m and meta["amount_sign"] == "DEBIT")
    meta["is_interest_credit_tx"]  = "interest" in disp and "credit" in disp or meta["amount_sign"] == "CREDIT"

    # statement interest numbers if present
    if domain == "statements":
        meta["interestCharged"]            = r.get("interestCharged", 0.0)
        meta["totalNonTrailingInterest"]   = r.get("totalNonTrailingInterest", r.get("nonTrailingInterest", 0.0))
        meta["totalTrailingInterest"]      = r.get("totalTrailingInterest", r.get("trailingInterest", 0.0))
        meta["isTrailingInterestApplied"]  = _bool(r.get("isTrailingInterestApplied"))
    return meta

# ---------- Add rows for each domain (core change you asked for) ----------

def _add_rows(docs: List[Document], domain: str, account_id: str, rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        meta = _base_meta(account_id, domain, r)

        if domain == "transactions":
            text = _render_transaction(r)
            key  = r.get("transactionId")
        elif domain == "payments":
            text = _render_payment(r)
            key  = r.get("paymentId")
        elif domain == "statements":
            text = _render_statement(r)
            key  = r.get("statementId")
        else:  # account_summary
            text = _render_account_summary(r)
            key  = r.get("accountId")

        meta["key"] = key

        # pack raw JSON at the end for full fidelity (helps debugging + retrieval)
        full_text = text + "\n\nRAW_JSON:\n" + json.dumps(r, ensure_ascii=False)

        docs.append(Document(text=full_text, metadata=meta))

def build_account_index(
    account_id: str,
    base_dir: Union[str, Path] = ACCOUNTS_DATA_DIR,
    persist_dir: Optional[Union[str, Path]] = None,
) -> BuildResult:
    """
    Build an account-specific FAISS index from JSON domain files, with rich
    text and metadata per row (transactions/payments/statements/account_summary).
    """
    _ensure_embed_model()

    if persist_dir is None:
        persist_dir = Path(str(ACCOUNTS_INDEX_DIR).format(account_id=account_id))
    else:
        persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    paths = _account_paths(account_id, base_dir)
    docs: List[Document] = []

    if paths["transactions"].exists():
        _add_rows(docs, "transactions", account_id, _rows(_read_json(paths["transactions"])))

    if paths["payments"].exists():
        _add_rows(docs, "payments", account_id, _rows(_read_json(paths["payments"])))

    if paths["statements"].exists():
        _add_rows(docs, "statements", account_id, _rows(_read_json(paths["statements"])))

    if paths["account_summary"].exists():
        _add_rows(docs, "account_summary", account_id, _rows(_read_json(paths["account_summary"])))

    if not docs:
        dim = len(Settings.embed_model.get_text_embedding("probe"))
        return BuildResult(count=0, persist_dir=str(persist_dir), dim=dim)

    storage, _ = _mk_storage_with_faiss(persist_dir)
    VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)
    storage.persist()

    dim = len(Settings.embed_model.get_text_embedding("probe"))
    return BuildResult(count=len(docs), persist_dir=str(persist_dir), dim=dim)


# ============================== Build both =====================================

def build_all(
    account_id: Optional[str] = None,
    accounts_base_dir: Union[str, Path] = ACCOUNTS_DATA_DIR,
    knowledge_dir: Union[str, Path] = KNOWLEDGE_DATA_DIR,
) -> Dict[str, Any]:
    _ensure_embed_model()
    out: Dict[str, Any] = {"index_root": str(INDEX_STORE_DIR)}

    try:
        kr = ensure_knowledge_index(knowledge_dir=knowledge_dir, persist_dir=KNOWLEDGE_INDEX_DIR)
        out["knowledge"] = {"count": kr.count, "persist_dir": kr.persist_dir, "dim": kr.dim}
    except Exception as e:
        out["knowledge"] = {"error": f"{type(e).__name__}: {e}"}

    if account_id:
        try:
            ar = build_account_index(account_id=account_id, base_dir=accounts_base_dir)
            out["account"] = {"count": ar.count, "persist_dir": ar.persist_dir, "dim": ar.dim}
        except Exception as e:
            out["account"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ============================== CLI ============================================

if __name__ == "__main__":
    """
    Example:
      python -m core.retrieval.index_builder                      # knowledge only
      python -m core.retrieval.index_builder 3b1ba69f-...-3617a   # also build account
    """
    import sys
    _ensure_embed_model()
    acct_id = sys.argv[1] if len(sys.argv) > 1 else None
    res = build_all(account_id=acct_id)
    print(json.dumps(res, indent=2))