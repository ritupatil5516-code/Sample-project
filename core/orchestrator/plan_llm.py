# core/orchestrator/plan_llm.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

CONFIG_FILE = "config/app.yaml"

def _read_config(path: str = CONFIG_FILE) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# Lightweight LLM client (OpenAI-compatible / Azure-compatible)
# -----------------------------------------------------------------------------

class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> Dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=headers, json=payload)
            # Log server error body if any
            if r.status_code >= 400:
                print("[LLM ERROR]", r.status_code, r.text)
            r.raise_for_status()
            return r.json()

def _build_llm(cfg: Dict[str, Any]) -> _LLMClient:
    llm_cfg = (cfg.get("llm") or {})
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
        api_base = "https://" + api_base
    api_key = (llm_cfg.get("api_key")
               or os.getenv(llm_cfg.get("api_key_env", "") or "OPENAI_API_KEY", "")
               or llm_cfg.get("key") or "").strip()
    model = (llm_cfg.get("model") or "gpt-4o-mini").strip()
    if not api_key:
        raise RuntimeError("LLM API key missing. Set llm.api_key or llm.api_key_env in config/app.yaml.")
    return _LLMClient(api_base=api_base, api_key=api_key, model=model)

# -----------------------------------------------------------------------------
# Planner contract (tools the model can pick)
# -----------------------------------------------------------------------------

# Domains and capabilities:
# - DSL (structured): transactions | payments | statements | account_summary
#   Ops: get_field, find_latest, sum_where, topk_by_sum, list_where, semantic_search
# - RAG (semantic): rag.unified_answer | rag.account_answer | rag.knowledge_answer

SYSTEM_CONTRACT = (
    "You are a planner for a credit-card copilot. "
    "Given a user question, you must output STRICT JSON with this schema:\n"
    "{\n"
    '  "intent": string,\n'
    '  "calls": [ {"domain_id": string, "capability": string, "args": object} ],\n'
    '  "must_produce": [],\n'
    '  "risk_if_missing": []\n'
    "}\n\n"
    "DOMAINS + OPS:\n"
    "- transactions: [get_field, find_latest, sum_where, topk_by_sum, list_where, semantic_search]\n"
    "- payments:     [get_field, find_latest, sum_where, topk_by_sum, list_where, semantic_search]\n"
    "- statements:   [get_field, find_latest, sum_where, topk_by_sum, list_where, semantic_search]\n"
    "- account_summary: [get_field, find_latest, list_where]\n"
    "- rag: [unified_answer, account_answer, knowledge_answer]\n\n"
    "ARG RULES (DSL):\n"
    "- For get_field: {account_id?: string, field: string} where field is a dotted path (e.g., "
    "'accountStatus', 'persons[0].ownershipType', 'period', 'currentBalance').\n"
    "- For find_latest: {account_id?: string, domain: one_of(transactions|payments|statements), "
    "  key: string (dotted path), where?: object, date_fields?: [strings]}\n"
    "- For sum_where: {account_id?: string, domain: string, where?: object, value_path?: string (default 'amount')}\n"
    "- For topk_by_sum: {account_id?: string, domain: string, key_field: string, where?: object, k?: int}\n"
    "- For list_where: {account_id?: string, domain: string, where?: object, limit?: int}\n"
    "- For semantic_search: {account_id?: string, domain: string, query: string, k?: int}\n\n"
    "RAG RULES:\n"
    "- unified_answer: prefer when the question may require both account JSONs and policy/handbook context, or when fuzzy.\n"
    "- account_answer: when the question is free-form but should be answered from the user's account JSONs only.\n"
    "- knowledge_answer: when the question is policy/handbook only.\n"
    "- RAG args: {account_id?: string, k?: int}\n\n"
    "ROUTING HINTS:\n"
    "- If the question asks directly for a specific field (e.g., 'what's my account status'), "
    "use account_summary.get_field with field='accountStatus'.\n"
    "- 'Where did I spend the most …' -> transactions.topk_by_sum (key_field='merchantName', set where/period if present).\n"
    "- 'Did I buy anything from <merchant>?' -> transactions.list_where with where={'merchantName': '<merchant>'} "
    "OR transactions.semantic_search with query including the merchant.\n"
    "- 'Why was I charged interest …' or policy questions -> rag.unified_answer.\n"
    "- When in doubt or the user is fuzzy, choose rag.unified_answer.\n\n"
    "OUTPUT STRICT JSON ONLY. No markdown, no commentary."
)

# A few short exemplars that teach the model the shape
EXAMPLES: List[Dict[str, Any]] = [
    {
        "intent": "account_status",
        "calls": [
            {"domain_id": "account_summary", "capability": "get_field",
             "args": {"field": "accountStatus"}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    {
        "intent": "spend_insights",
        "calls": [
            {"domain_id": "transactions", "capability": "topk_by_sum",
             "args": {"key_field": "merchantName", "k": 5, "where": {"transactionType": "DEBIT"}}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    {
        "intent": "merchant_lookup",
        "calls": [
            {"domain_id": "transactions", "capability": "list_where",
             "args": {"where": {"merchantName": "apple"}, "limit": 20}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    {
        "intent": "why_interest",
        "calls": [
            {"domain_id": "rag", "capability": "unified_answer", "args": {}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    }
]

def _examples_block() -> str:
    return "\n".join(json.dumps(e, separators=(",", ":")) for e in EXAMPLES)

# -----------------------------------------------------------------------------
# Message building / parsing helpers
# -----------------------------------------------------------------------------

def _coerce_chat_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure message content is valid (OpenAI format)."""
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, (str, list)):
            out.append({"role": role, "content": content})
        elif isinstance(content, dict):
            inner = content.get("content")
            out.append({"role": role, "content": inner if isinstance(inner, str) else json.dumps(content, ensure_ascii=False)})
        else:
            out.append({"role": role, "content": str(content)})
    return out

def _extract_outer_json(text: str) -> str:
    """Grab the outermost JSON object from the response text."""
    if not isinstance(text, str):
        return "{}"
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end+1] if (start >= 0 and end >= 0 and end > start) else "{}"

def _normalize_plan(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}
    obj.setdefault("intent", "unknown")
    safe_calls: List[Dict[str, Any]] = []
    for c in (obj.get("calls") or []):
        if not isinstance(c, dict):
            continue
        c.setdefault("domain_id", "")
        c.setdefault("capability", "")
        args = c.get("args")
        c["args"] = args if isinstance(args, dict) else {}
        safe_calls.append(c)
    obj["calls"] = safe_calls
    obj.setdefault("must_produce", [])
    obj.setdefault("risk_if_missing", [])
    return obj

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def make_plan(question: str, config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Builds a plan by calling the LLM with the contract + examples.
    Returns a dict: {intent, calls, must_produce, risk_if_missing}.
    """
    cfg = _read_config(config_path)
    llm = _build_llm(cfg)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_CONTRACT},
        {"role": "user", "content":
            "Return ONLY JSON. Do not include explanations or code fences.\n\n"
            f"Examples:\n{_examples_block()}\n\n"
            f"Question: {question}"
        }
    ]
    messages = _coerce_chat_messages(messages)

    resp = llm.chat(messages, temperature=0.0)
    raw = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
    json_block = _extract_outer_json(raw)

    try:
        obj = json.loads(json_block)
    except Exception:
        obj = {}

    return _normalize_plan(obj)

# Backward-compat alias
def llm_plan(question: str) -> Dict[str, Any]:
    return make_plan(question, CONFIG_FILE)

# Quick manual test
if __name__ == "__main__":
    q = "what's my account status?"
    print(make_plan(q))