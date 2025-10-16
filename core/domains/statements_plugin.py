# core/domains/statements_plugin.py
from __future__ import annotations
from typing import Any, Dict, List
from .base import Domain
from core.orchestrator.dsl_ops import op_get_field, op_list_where, op_sum_where, op_find_latest
from domains.statements.loader import load_statements

ALIASES = {
    "date": ["closingDateTime", "openingDateTime", "period"],
    "period": ["period"],
    "interest": ["interestCharged", "interest_total"],
    "amount": ["amount", "balance"],
}

def _load(account_id: str | None, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return load_statements(account_id=account_id, cfg=cfg)

OPS = {
    "get_field": op_get_field,
    "list_where": op_list_where,
    "sum_where": op_sum_where,
    "find_latest": op_find_latest,
}

DOMAIN = Domain(id="statements", load=_load, ops=OPS, aliases=ALIASES)