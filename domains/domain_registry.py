# domains/registry.py (Simple Mode)
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

# --------- minimal plugin meta holder ---------
class PluginMeta:
    def __init__(self, aliases: Dict[str, str], is_list: bool, ts_keys: List[str]):
        self.ALIASES = aliases or {}
        self.is_list = bool(is_list)
        self.timestamp_keys = list(ts_keys or [])

# --------- canonical domain ids ---------
_NORMALIZE: Dict[str, str] = {
    "account": "accounts",
    "account_summary": "accounts",
    "acct": "accounts",
    "accts": "accounts",
    "txn": "transactions",
    "txns": "transactions",
    "payment": "payments",
    "statement": "statements",
}
def normalize_domain(d: str) -> str:
    if not d: return d
    d = d.strip().lower().replace("-", "_")
    return _NORMALIZE.get(d, d)

# --------- single-source metadata here ---------
PLUGINS: Dict[str, PluginMeta] = {
    "accounts": PluginMeta(
        aliases={
            "status": "accountStatus",
            "account_status": "accountStatus",
            "balance": "currentBalance",
            "current_balance": "currentBalance",
            "available": "availableCredit",
            "available_credit": "availableCredit",
            "credit_limit": "creditLimit",
        },
        is_list=False,
        ts_keys=["updatedAt", "date"],
    ),
    "transactions": PluginMeta(
        aliases={
            "date": "postedDateTime",
            "status": "transactionStatus",
            "type": "displayTransactionType",
            "merchant": "merchantName",
            "amount": "amount",
            "category": "category",
        },
        is_list=True,
        ts_keys=["postedDateTime", "transactionDateTime", "date"],
    ),
    "payments": PluginMeta(
        aliases={
            "date": "paymentPostedDateTime",
            "amount": "amount",
            "status": "status",
        },
        is_list=True,
        ts_keys=["paymentPostedDateTime", "paymentDateTime", "date"],
    ),
    "statements": PluginMeta(
        aliases={
            "period": "period",
            "close": "closingDateTime",
            "interest": "interestCharged",
            "balance": "statementBalance",
        },
        is_list=True,
        ts_keys=["closingDateTime", "openingDateTime", "period", "date"],
    ),
}

def get_plugin(domain: str) -> Optional[PluginMeta]:
    return PLUGINS.get(normalize_domain(domain))

# --------- loaders (your existing ones) ---------
try:
    from domains.transactions.loader import load_transactions
except Exception:
    def load_transactions(account_id: Optional[str], cfg: Dict[str, Any]): return []

try:
    from domains.payments.loader import load_payments
except Exception:
    def load_payments(account_id: Optional[str], cfg: Dict[str, Any]): return []

try:
    from domains.statements.loader import load_statements
except Exception:
    def load_statements(account_id: Optional[str], cfg: Dict[str, Any]): return []

try:
    from domains.account_summary.loader import load_account_summary
except Exception:
    def load_account_summary(account_id: Optional[str], cfg: Dict[str, Any]): return {}

LOADERS: Dict[str, Callable[[Optional[str], Dict[str, Any]], Any]] = {
    "transactions": load_transactions,
    "payments": load_payments,
    "statements": load_statements,
    "accounts": load_account_summary,
}

def get_loader(domain: str):
    return LOADERS.get(normalize_domain(domain))

def load_domain(domain: str, account_id: Optional[str], cfg: Dict[str, Any]):
    fn = get_loader(domain)
    return fn(account_id, cfg) if fn else None

# handy helpers (optional)
def alias(domain: str, field: str) -> str:
    p = get_plugin(domain)
    return (p.ALIASES if p else {}).get(field, field)

def is_list(domain: str) -> bool:
    p = get_plugin(domain)
    return bool(p and p.is_list)

def timestamp_keys(domain: str) -> List[str]:
    p = get_plugin(domain)
    return list(p.timestamp_keys) if p else []

def available_domains() -> List[str]:
    return sorted(PLUGINS.keys())