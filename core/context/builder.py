# core/context/builder.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

PACK_PATH = Path("core/context/packs/core.yaml")

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _as_block(value: Any) -> str:
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

def build_context(
    intent: str,
    question: Optional[str],
    plan: Optional[Dict[str, Any]] = None,
    *,
    pack_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Works with two shapes of core.yaml:
      1) Full pack with top-level: system/glossary/reasoning/planner_contract/domains/extras
      2) Your current pack with only: planner_rules: {synonyms, defaults, routes}
    """
    pack_file = Path(pack_path or PACK_PATH)
    pack = _load_yaml(pack_file)

    msgs: List[Dict[str, str]] = []

    # ----- Preferred: use top-level content if present
    system_text  = (pack.get("system") or "") if isinstance(pack, dict) else ""
    glossary     = pack.get("glossary")
    reasoning    = pack.get("reasoning")
    contract     = pack.get("planner_contract")
    domains      = pack.get("domains")
    extras       = pack.get("extras")

    if glossary:  msgs.append(_section_msg("Glossary", glossary))
    if reasoning: msgs.append(_section_msg("Reasoning", reasoning))
    if intent == "planner" and contract:
        msgs.append(_section_msg("Planner Contract", contract))
    if domains:   msgs.append(_section_msg("Domains", domains))
    if extras:    msgs.append(_section_msg("Notes", extras))

    # ----- Fallback: derive useful context from planner_rules when top-level is missing
    pr = (pack.get("planner_rules") or {}) if isinstance(pack, dict) else {}
    if not system_text:
        system_text = (
            "You are the planner for a Smart Finance Copilot.\n"
            "Decide a single call over domains: transactions, payments, statements, accounts.\n"
            "Prefer deterministic operations (get_field, find_latest, sum_where, topk_by_sum, list_where).\n"
            "If the question asks WHY/EXPLAIN or mentions policy/handbook/fees/interest policy, set strategy=rag:unified."
        )

    syn = pr.get("synonyms") or {}
    if syn:
        msgs.append(_section_msg("Synonyms (routing hints)", syn))

    defaults = pr.get("defaults") or {}
    if defaults:
        msgs.append(_section_msg("Default time interpretations", defaults))

    # Provide a compact contract if none existed
    if intent == "planner" and not contract:
        minimal_contract = (
            "Return ONLY a JSON object of the form:\n"
            "{\n"
            '  \"intent\": \"string\",\n'
            '  \"calls\": [\n'
            '    {\"domain_id\":\"transactions|payments|statements|accounts\",'
            '     \"capability\":\"get_field|find_latest|sum_where|topk_by_sum|list_where|semantic_search|compare_periods\",'
            '     \"args\": {...}, \"strategy\":\"deterministic|rag:unified|rag:knowledge|auto\"}\n'
            '  ],\n'
            '  \"must_produce\": [],\n'
            '  \"risk_if_missing\": [],\n'
            '  \"strategy\": \"deterministic|rag:unified|rag:knowledge|auto\"\n'
            "}\n"
            "- Domains allowed: transactions, payments, statements, accounts.\n"
            "- Use statements.closingDateTime / transactions.postedDateTime / payments.paymentPostedDateTime for recency."
        )
        msgs.append({"role": "system", "content": minimal_contract})

    # clean blanks
    msgs = [m for m in msgs if m]

    return {
        "system": system_text.strip(),
        "context_msgs": msgs,
    }