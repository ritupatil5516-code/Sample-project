# core/domains/payments_plugin.py
from __future__ import annotations
from typing import Any, Dict, List
from .base import Domain
from core.orchestrator.dsl_ops import op_get_field, op_list_where, op_sum_where, op_find_latest
from domains.payments.loader import load_payments

ALIASES = {
    "date": ["paymentPostedDateTime", "paymentDateTime"],
    "status": ["paymentStatus", "status"],
    "amount": ["paymentAmount", "amount"],
    "description": ["description"],
}

def _load(account_id: str | None, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return load_payments(account_id=account_id, cfg=cfg)

OPS = {
    "get_field": op_get_field,
    "list_where": op_list_where,
    "sum_where": op_sum_where,
    "find_latest": op_find_latest,
}

DOMAIN = Domain(id="payments", load=_load, ops=OPS, aliases=ALIASES)