# core/domains/account_summary_plugin.py
from __future__ import annotations
from typing import Any, Dict
from .base import Domain
from core.orchestrator.dsl_ops import op_get_field
from domains.account_summary.loader import load_account_summary

ALIASES = {
    "status": ["accountStatus", "status"],
    "availableCredit": ["availableCreditLimit", "availableCredit"],
    "currentBalance": ["currentBalance", "currentAdjustedBalance"],
    "creditLimit": ["creditLimit"],
}

def _load(account_id: str | None, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return load_account_summary(account_id=account_id, cfg=cfg)

OPS = {
    "get_field": op_get_field,
}

# Expose as "accounts" (executor will normalize)
DOMAIN = Domain(id="accounts", load=_load, ops=OPS, aliases=ALIASES)