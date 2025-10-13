# core/orchestrator/plan_llm.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx
import yaml

from core.context.builder import build_context
from core.context.hints import build_hint_for_question

# -----------------------------------------------------------------------------#
# Config + message helpers
# -----------------------------------------------------------------------------#

def _read_app_cfg() -> Dict[str, Any]:
    p = Path("config/app.yaml")
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _coerce_chat_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "system")
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            out.append({"role": role, "content": content})
        elif isinstance(content, dict):
            inner = content.get("content") if "content" in content else None
            if isinstance(inner, str):
                out.append({"role": role, "content": inner})
            else:
                out.append({"role": role, "content": json.dumps(content, ensure_ascii=False)})
        else:
            out.append({"role": role, "content": str(content)})
    return out

# -----------------------------------------------------------------------------#
# Minimal OpenAI-compatible LLM client for planning (keeps your old style)
# -----------------------------------------------------------------------------#

class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, messages: List[Dict[str, Any]], *, model: Optional[str] = None, temperature: float = 0.0) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": float(temperature),
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    print("[LLM ERROR]", r.status_code, r.text)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Planner LLM error: {e.response.text}") from e

def build_llm_from_config(cfg: Dict[str, Any]) -> _LLMClient:
    llm_cfg = (cfg.get("llm") or {})
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
        api_base = "https://" + api_base
    api_key = (llm_cfg.get("api_key")
               or (llm_cfg.get("api_key_env") and os.getenv(llm_cfg.get("api_key_env"), ""))
               or llm_cfg.get("key") or "")
    model = (llm_cfg.get("model") or "gpt-4o-mini").strip()
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing LLM API key. Set llm.api_key or llm.api_key_env in config/app.yaml.")
    return _LLMClient(api_base=api_base, api_key=api_key, model=model)

# -----------------------------------------------------------------------------#
# Field map (lets planner emit account_summary.get_field)
# -----------------------------------------------------------------------------#

ACCOUNT_SUMMARY_FIELDS: Dict[str, List[str]] = {
    "accountStatus": ["status", "account status", "state", "account_state"],
    "currentBalance": ["current balance", "balance", "statementBalance", "currentAdjustedBalance"],
    "availableCreditAmount": ["available credit", "availableCredit", "available credit amount"],
    "creditLimit": ["credit limit"],
    "minimumDueAmount": ["minimum due", "minimum payment due"],
    "paymentDueDate": ["due date", "payment due date", "next due date"],
    "accountNumberLast4": ["last 4", "last4", "account last 4"],
}

# -----------------------------------------------------------------------------#
# Planner contract + examples
# -----------------------------------------------------------------------------#

SYSTEM_CONTRACT = (
  "You are a planner for a credit-card copilot. "
  "Given a user question, output STRICT JSON only: "
  "{intent: string, calls: [{domain_id, capability, args}], must_produce:[], risk_if_missing:[]}.\n"
  "Domains/capabilities:\n"
  "- transactions: {last_transaction, top_merchants, average_per_month, spend_in_period, "
  "  list_over_threshold, purchases_in_cycle, semantic_search, find_by_merchant, get_field}\n"
  "- payments: {last_payment, total_credited_year, payments_in_period, get_field}\n"
  "- statements: {total_interest, interest_breakdown, trailing_interest, get_field}\n"
  "- account_summary: {current_balance, available_credit, get_field}\n"
  "- rag: {account_answer, knowledge_answer}\n"
  "Guidance:\n"
  "- If the user asks for a literal field or clear synonym in account profile "
  "  (e.g., 'account status', 'credit limit', 'due date'), use account_summary.get_field "
  "  with args.key_path set to the CANONICAL camelCase key (e.g., 'accountStatus', 'creditLimit', 'paymentDueDate').\n"
  "- Use transactions/payments/statements.get_field for list-style field retrieval; you may include "
  "  args.filter (equals or contains with '~'), args.sort_by, args.order ('asc'|'desc'), args.limit, args.agg.\n"
  "- If the user asks for last/latest/recent interest or omits period, set args.period = null; "
  "  if they say 'last interest' or 'most recent interest charged', also set args.nonzero = true.\n"
  "- 'Where did I spend the most ...' -> transactions.top_merchants (add args.period if a time frame is given; e.g., 'LAST_12M').\n"
  "- 'Did I buy anything from <merchant>?' -> transactions.find_by_merchant with args.merchant_query='<merchant>'.\n"
  "- Use transactions.semantic_search for fuzzy concepts (e.g., args.query='travel purchases').\n"
  "- Use rag.account_answer for open-ended, fuzzy, multi-field, or summarization questions about the user's account data.\n"
  "- Use rag.knowledge_answer for policy/handbook/general documentation questions.\n"
  "Return ONLY JSON, no commentary."
)

try:
    from core.orchestrator.intent_examples import EXAMPLES as _EXTERNAL_EXAMPLES  # optional
    EXAMPLES: List[Dict[str, Any]] = _EXTERNAL_EXAMPLES
except Exception:
    EXAMPLES = [
        {"intent":"get_current_balance","calls":[{"domain_id":"account_summary","capability":"current_balance","args":{}}],"must_produce":[],"risk_if_missing":[]},
        {"intent":"account_status","calls":[{"domain_id":"account_summary","capability":"get_field","args":{"key_path":"accountStatus"}}],"must_produce":[],"risk_if_missing":[]},
        {"intent":"credit_limit","calls":[{"domain_id":"account_summary","capability":"get_field","args":{"key_path":"creditLimit"}}],"must_produce":[],"risk_if_missing":[]},
        {"intent":"due_date","calls":[{"domain_id":"account_summary","capability":"get_field","args":{"key_path":"paymentDueDate"}}],"must_produce":[],"risk_if_missing":[]},
        {"intent":"last_transaction","calls":[{"domain_id":"transactions","capability":"last_transaction","args":{}}],"must_produce":[],"risk_if_missing":[]},
        {"intent":"top_merchants_last_year","calls":[{"domain_id":"transactions","capability":"top_merchants","args":{"period":"LAST_12M"}}],"must_produce":[],"risk_if_missing":[]},
        {"intent":"why_interest","calls":[
            {"domain_id":"statements","capability":"total_interest","args":{"period":None,"nonzero":True}},
            {"domain_id":"statements","capability":"interest_breakdown","args":{"period":None,"nonzero":True}},
            {"domain_id":"statements","capability":"trailing_interest","args":{"period":None}}
        ],"must_produce":[],"risk_if_missing":[]},
        {"intent":"explain_spending_pattern","calls":[{"domain_id":"rag","capability":"account_answer","args":{"scope":"account"}}],"must_produce":[],"risk_if_missing":[]},
        {"intent":"policy_question","calls":[{"domain_id":"rag","capability":"knowledge_answer","args":{"scope":"knowledge"}}],"must_produce":[],"risk_if_missing":[]},
    ]

def _examples_block() -> str:
    return "\n".join(json.dumps(e, separators=(",", ":")) for e in EXAMPLES)

# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#

def llm_plan(question: str) -> Dict[str, Any]:
    cfg = _read_app_cfg()
    llm = build_llm_from_config(cfg)

    ctx = build_context(intent="planner", question=question, plan=None)
    hint = build_hint_for_question(question)
    hint_msg = [{"role": "system", "content": hint}] if hint else []

    # Tell the model your canonical keys + synonyms (helps it pick get_field)
    field_help = {
        "domain": "account_summary",
        "canonical_keys": list(ACCOUNT_SUMMARY_FIELDS.keys()),
        "synonyms": ACCOUNT_SUMMARY_FIELDS,
    }
    field_help_msg = [{"role": "system", "content": f"Field map: {json.dumps(field_help)}"}]

    messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": ctx.get("system", "")}]
        + (ctx.get("context_msgs") or [])
        + hint_msg
        + field_help_msg
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
    messages = _coerce_chat_messages(messages)

    raw = llm.complete(messages, model=(cfg.get("llm") or {}).get("model"), temperature=0)
    start = raw.find("{"); end = raw.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        raw = raw[start:end + 1]

    try:
        obj = json.loads(raw)
    except Exception:
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

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