# core/domains/registry.py
from __future__ import annotations
import importlib
from typing import Dict, List
from .base import Domain

_DOMAIN_MODULES = [
    "core.domains.transactions_plugin",
    "core.domains.payments_plugin",
    "core.domains.statements_plugin",
    "core.domains.account_summary_plugin",  # planner may say "accounts" → executor normalizes
]

class DomainRegistry:
    def __init__(self):
        self._by_id: Dict[str, Domain] = {}
        for modname in _DOMAIN_MODULES:
            mod = importlib.import_module(modname)
            dom: Domain = getattr(mod, "DOMAIN")
            self._by_id[dom.id] = dom

    def get(self, dom_id: str) -> Domain | None:
        return self._by_id.get(dom_id)

    @property
    def domains(self) -> List[Domain]:
        return list(self._by_id.values())

    def contract(self) -> str:
        parts = []
        for d in self.domains:
            caps = ",".join(sorted(d.ops.keys()))
            parts.append(f"{d.id}:{{{caps}}}")
        return " ; ".join(parts)

REGISTRY = DomainRegistry()