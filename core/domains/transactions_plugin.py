# core/domains/transactions_plugin.py
from __future__ import annotations
from typing import Any, Dict, List
from .base import Domain
from core.orchestrator.dsl_ops import (
    op_get_field, op_list_where, op_sum_where, op_topk_by_sum, op_find_latest, op_semantic_search
)
from domains.transactions.loader import load_transactions

ALIASES = {
    "date": ["transactionDateTime", "postedDateTime", "authDateTime"],
    "status": ["transactionStatus"],
    "type": ["displayTransactionType", "transactionType"],
    "merchantName": ["merchantName", "description"],
    "amount": ["amount"],
    "category": ["category"],
}

def _load(account_id: str | None, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return load_transactions(account_id=account_id, cfg=cfg)

OPS = {
    "get_field": op_get_field,
    "list_where": op_list_where,
    "sum_where": op_sum_where,
    "topk_by_sum": op_topk_by_sum,
    "find_latest": op_find_latest,
    "semantic_search": op_semantic_search,  # optional (needs faiss_registry)
}

DOMAIN = Domain(id="transactions", load=_load, ops=OPS, aliases=ALIASES)