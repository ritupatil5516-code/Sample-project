# core/context/builder.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


"""
builder.py
----------
Builds prompt context for the LLM from ONE unified pack:
  core/context/packs/core.yaml

Expected sections inside core.yaml (all optional):
  system: str                     # main system instruction
  glossary: list[str] | dict      # key terms
  reasoning: list[str] | dict     # nudges like "last txn = max(timestamp)"
  planner_contract: str | dict    # how to emit the plan JSON
  domains: list[str] | dict       # visible domains (informational)
  extras: list[str] | dict        # any additional notes you want included

Usage:
  ctx = build_context(intent="planner", question=..., plan=None)
  messages = [
      {"role":"system","content": ctx["system"]},
      *ctx["context_msgs"],
      # then your specific planner SYSTEM_CONTRACT + the user message...
  ]
"""


# ------------------------------- helpers --------------------------------------

def _safe_yaml_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # malformed YAML → ignore gracefully
        return {}


def _as_block(value: Any) -> str:
    """Normalize lists/dicts/strings into a readable block."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {str(x)}" for x in value)
    if isinstance(value, dict):
        lines: List[str] = []
        for k, v in value.items():
            if isinstance(v, (list, dict)):
                dumped = yaml.safe_dump(v, sort_keys=False).strip()
                lines.append(f"{k}:\n{dumped}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
    return str(value)


def _section_msg(title: str, content: Any) -> Dict[str, str]:
    body = _as_block(content).strip()
    if not body:
        return {}
    return {"role": "system", "content": f"{title}:\n{body}"}


# ------------------------------- main API -------------------------------------

def build_context(
    intent: str,
    question: Optional[str],
    plan: Optional[Dict[str, Any]] = None,
    *,
    pack_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build context messages for the given intent.
    - intent: "planner" | "answer" | anything else (used only to decide which sections to include)
    - question/plan: not embedded here; caller adds the user message and any plan trace
    - pack_path: override path to the single pack (default core/context/packs/core.yaml)

    Returns:
      {
        "system": str,                 # main system string (may be empty)
        "context_msgs": List[Msg],     # system messages for glossary/reasoning/contract/etc.
      }
    """
    pack_file = Path(pack_path or "core/context/packs/core.yaml")
    pack = _safe_yaml_load(pack_file)

    # Extract canonical sections (all optional)
    system_text        = pack.get("system", "") or ""
    glossary_section   = pack.get("glossary")
    reasoning_section  = pack.get("reasoning")
    contract_section   = pack.get("planner_contract")
    domains_section    = pack.get("domains")
    extras_section     = pack.get("extras")

    msgs: List[Dict[str, str]] = []

    # Order matters: glossary → reasoning → (planner contract for planner intent) → domains → extras
    if glossary_section:
        msg = _section_msg("Glossary", glossary_section)
        if msg: msgs.append(msg)

    if reasoning_section:
        msg = _section_msg("Reasoning", reasoning_section)
        if msg: msgs.append(msg)

    # Only include planner contract when composing the planning prompt
    if intent == "planner" and contract_section:
        msg = _section_msg("Planner Contract", contract_section)
        if msg: msgs.append(msg)

    if domains_section:
        msg = _section_msg("Domains", domains_section)
        if msg: msgs.append(msg)

    if extras_section:
        msg = _section_msg("Notes", extras_section)
        if msg: msgs.append(msg)

    return {
        "system": system_text.strip(),
        "context_msgs": msgs,
    }