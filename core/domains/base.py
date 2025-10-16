# core/domains/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

class OpFn(Protocol):
    def __call__(self, data: Any, args: Dict[str, Any], ctx: "OpContext") -> Dict[str, Any]: ...

@dataclass
class OpContext:
    account_id: Optional[str]
    session_id: str
    cfg: Dict[str, Any]
    scratch: Dict[str, Any]
    trace: bool = True

@dataclass
class Domain:
    id: str
    load: Callable[[Optional[str], Dict[str, Any]], Any]
    ops: Dict[str, OpFn]
    aliases: Dict[str, List[str]]
    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "capabilities": sorted(self.ops.keys())}