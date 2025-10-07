# core/orchestrator/plan_llm.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx
import yaml

from core.context.builder import build_context
from core.context.hints import build_hint_for_question

import os, httpx, yaml, json, traceback

cfg = yaml.safe_load(open("config/app.yaml").read())
key_env = (cfg.get("llm", {}).get("api_key_env") or "OPENAI_API_KEY").strip()
key = (os.getenv(key_env) or "").strip()
base = (cfg.get("llm", {}).get("base_url") or "https://api.openai.com/v1").strip()

print("[PLANNER DEBUG] base_url =", base)
print("[PLANNER DEBUG] key_env =", key_env, "present?" , bool(key))
# ------------------------------ config helpers --------------------------------

def _read_app_cfg() -> Dict[str, Any]:
    cfg_path = Path("config/app.yaml")
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


# ------------------------------- LLM wrapper ----------------------------------

class _LLMClient:
    """
    Minimal chat-completions client over HTTP.
    Default: OpenAI-compatible (gpt-4o-mini).
    Later you can swap api_base/model/key in config to point at Llama 70B.
    """
    def __init__(self, api_base: str, api_key: str, model: str, timeout: float = 30.0):
        # normalize base to include scheme
        if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
            api_base = "https://" + api_base
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or "gpt-4o-mini"
        self.timeout = timeout

    def complete(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": float(temperature),
        }
        with httpx.Client(base_url=self.api_base, timeout=self.timeout) as client:
            r = client.post("/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


def build_llm_from_config(cfg: Dict[str, Any]) -> _LLMClient:
    llm_cfg = (cfg.get("llm") or {})
    api_base = llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1"
    api_key  = llm_cfg.get("api_key") or llm_cfg.get("key") or ""
    model    = llm_cfg.get("model") or "gpt-4o-mini"
    return _LLMClient(api_base=api_base, api_key=api_key, model=model)


# ------------------------------ Planning prompt -------------------------------

SYSTEM_CONTRACT = (
    "You are a planner for a credit-card copilot.\n"
    "Given the user's question, output a STRICT JSON object with keys:\n"
    "{\n"
    '  "intent": string,\n'
    '  "calls": [ { "domain_id": string, "capability": string, "args": object } ],\n'
    '  "must_produce": [],\n'
    '  "risk_if_missing": []\n'
    "}\n"
    "Only include fields listed above. No comments, no extra keys, no markdown.\n"
    "\n"
    "Available domains & capabilities:\n"
    "- transactions: { spend_in_period, list_over_threshold, purchases_in_cycle, last_transaction, average_per_month, compare_periods }\n"
    "- payments:     { last_payment, total_credited_year, payments_in_period }\n"
    "- statements:   { total_interest, interest_breakdown, trailing_interest }\n"
    "- account_summary: { current_balance, available_credit }\n"
    "\n"
    "Conventions:\n"
    "- Periods may be 'YYYY' or 'YYYY-MM'. If user says latest/last/recent, set args.period to null.\n"
    "- For phrases like 'last interest' or 'most recent interest charged', set args.nonzero=true when relevant.\n"
    "- Keep args minimal and infer obvious values from the question.\n"
)

# A few tiny patterns so the model learns tool selection quickly.
EXAMPLES: List[Dict[str, Any]] = [
    {
        "intent": "last_transaction",
        "calls": [
            {"domain_id": "transactions", "capability": "last_transaction", "args": {}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    {
        "intent": "get_current_balance",
        "calls": [
            {"domain_id": "account_summary", "capability": "current_balance", "args": {}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    {
        "intent": "why_interest",
        "calls": [
            {"domain_id": "statements",  "capability": "total_interest",      "args": {"period": None, "nonzero": True}},
            {"domain_id": "statements",  "capability": "interest_breakdown",  "args": {"period": None, "nonzero": True}},
            {"domain_id": "statements",  "capability": "trailing_interest",   "args": {"period": None}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    {
        "intent": "spend_in_period",
        "calls": [
            {"domain_id": "transactions", "capability": "spend_in_period", "args": {"period": "2025-04"}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    {
        "intent": "average_per_month",
        "calls": [
            {"domain_id": "transactions", "capability": "average_per_month", "args": {"period": "2025"}}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
]


def _examples_block() -> str:
    # One compact string the model can see in the prompt
    return "\n".join(json.dumps(e, separators=(",", ":")) for e in EXAMPLES)


# --------------------------------- Public API ---------------------------------

def llm_plan(question: str) -> Dict[str, Any]:
    """
    Build a plan by calling the LLM with context packs + routing hints + contract.
    Returns a dict with defaults if parsing fails.
    """
    cfg = _read_app_cfg()
    llm = build_llm_from_config(cfg)

    # contextual system packs (planner mode)
    ctx = build_context(intent="planner", question=question, plan=None)

    # tiny routing hint (optional, only when pattern matches)
    hint = build_hint_for_question(question)
    hint_msg = [{"role": "system", "content": hint}] if hint else []

    messages: List[Dict[str, str]] = (
        [{"role": "system", "content": ctx["system"]}]
        + ctx.get("context_msgs", [])
        + hint_msg
        + [{"role": "system", "content": SYSTEM_CONTRACT}]
        + [{
            "role": "user",
            "content": (
                "Return ONLY JSON. Do not include explanations or markdown code fences.\n\n"
                f"Examples:\n{_examples_block()}\n\n"
                f"Question: {question}"
            )
        }]
    )

    raw = llm.complete(messages, model=(cfg.get("llm") or {}).get("model"), temperature=0)

    # Strip to the outermost JSON object if the model wrapped it in text
    start = raw.find("{"); end = raw.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        raw = raw[start:end + 1]

    try:
        obj = json.loads(raw)
    except Exception:
        # Last-resort default
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    # Safety: fill defaults and ensure args objects exist
    if not isinstance(obj, dict):
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    obj.setdefault("intent", "unknown")
    calls = obj.get("calls") or []
    safe_calls: List[Dict[str, Any]] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        c.setdefault("domain_id", "")
        c.setdefault("capability", "")
        c.setdefault("args", {})
        if not isinstance(c["args"], dict):
            c["args"] = {}
        safe_calls.append(c)
    obj["calls"] = safe_calls
    obj.setdefault("must_produce", [])
    obj.setdefault("risk_if_missing", [])

    return obj