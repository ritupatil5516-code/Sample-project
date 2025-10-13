# core/orchestrator/plan_llm.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx
import yaml

# Optional context packs / routing hints (keep your existing modules)
from core.context.builder import build_context
from core.context.hints import build_hint_for_question

# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def _read_app_cfg() -> Dict[str, Any]:
    """
    Reads config/app.yaml if present. Returns {} on any error.
    """
    p = Path("config/app.yaml")
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _coerce_chat_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure every message has a valid OpenAI-compatible 'content' value:
    - str -> keep
    - list (array-of-parts) -> keep
    - dict with 'content' -> unwrap if it's a str; else JSON-serialize
    - None -> drop
    - everything else -> str(...)
    """
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

# -----------------------------------------------------------------------------
# Minimal HTTP client (works with OpenAI-compatible APIs)
# -----------------------------------------------------------------------------

class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, messages: List[Dict[str, Any]], *, model: Optional[str] = None, temperature: float = 0.0) -> str:
        """
        Calls /chat/completions. Returns assistant message content (str).
        Raises with server error text for quick diagnosis in dev.
        """
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
                    # Print raw text so you see the exact server error in logs
                    print("[LLM ERROR]", r.status_code, r.text)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Planner LLM error: {e.response.text}") from e

def build_llm_from_config(cfg: Dict[str, Any]) -> _LLMClient:
    """
    Builds the LLM client from config/app.yaml.
    Supports OpenAI-compatible endpoints.
    """
    llm_cfg = (cfg.get("llm") or {})
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
        api_base = "https://" + api_base
    api_key = (llm_cfg.get("api_key") or
               (llm_cfg.get("api_key_env") and os.getenv(llm_cfg.get("api_key_env"), "")) or
               llm_cfg.get("key") or "")
    model = (llm_cfg.get("model") or "gpt-4o-mini").strip()

    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing LLM API key. Set llm.api_key or llm.api_key_env in config/app.yaml.")

    return _LLMClient(api_base=api_base, api_key=api_key, model=model)

# -----------------------------------------------------------------------------
# Field map (helps the planner emit account_summary.get_field)
# -----------------------------------------------------------------------------

ACCOUNT_SUMMARY_FIELDS: Dict[str, List[str]] = {
    "accountStatus": ["status", "account status", "state", "account_state"],
    "currentBalance": ["current balance", "balance", "statementBalance", "currentAdjustedBalance"],
    "availableCreditAmount": ["available credit", "availableCredit", "available credit amount"],
    "creditLimit": ["credit limit"],
    "minimumDueAmount": ["minimum due", "minimum payment due"],
    "paymentDueDate": ["due date", "payment due date", "next due date"],
    "accountNumberLast4": ["last 4", "last4", "account last 4"],
}

# -----------------------------------------------------------------------------
# Contract + Examples
# -----------------------------------------------------------------------------

SYSTEM_CONTRACT = (
  "You are a planner for a credit-card copilot. "
  "Given a user question, output STRICT JSON only: "
  "{intent: string, calls: [{domain_id, capability, args}], must_produce:[], risk_if_missing:[]}.\n"
  "Domains/capabilities:\n"
  "- transactions: {last_transaction, top_merchants, average_per_month, spend_in_period, "
  "  list_over_threshold, purchases_in_cycle, semantic_search, find_by_merchant}\n"
  "- payments: {last_payment, total_credited_year, payments_in_period}\n"
  "- statements: {total_interest, interest_breakdown, trailing_interest}\n"
  "- account_summary: {current_balance, available_credit, get_field}\n"
  "Guidance:\n"
  "- If the user asks for a literal field or clear synonym in account profile "
  "  (e.g., 'account status', 'credit limit', 'due date'), use account_summary.get_field "
  "  with args.key_path set to the CANONICAL camelCase key (e.g., 'accountStatus', 'creditLimit', 'paymentDueDate').\n"
  "- If the user asks for last/latest/recent interest or omits period, set args.period = null; "
  "  if they say 'last interest' or 'most recent interest charged', also set args.nonzero = true.\n"
  "- 'Where did I spend the most ...' -> transactions.top_merchants (add args.period if time frame is given; e.g., 'LAST_12M').\n"
  "- 'Did I buy anything from <merchant>?' -> transactions.find_by_merchant with args.merchant_query='<merchant>'.\n"
  "- Use transactions.semantic_search for fuzzy concepts (e.g., args.query='travel purchases').\n"
  "Return ONLY JSON, no commentary."
)

# You can keep examples in this file or import them if you created a separate module.
try:
    from core.orchestrator.intent_examples import EXAMPLES as _EXTERNAL_EXAMPLES  # optional
    EXAMPLES: List[Dict[str, Any]] = _EXTERNAL_EXAMPLES
except Exception:
    EXAMPLES = [
        # Essentials (keep it small; the contract already does heavy lifting)
        {
            "intent": "get_current_balance",
            "calls": [{"domain_id": "account_summary", "capability": "current_balance", "args": {}}],
            "must_produce": [], "risk_if_missing": []
        },
        {
            "intent": "account_status",
            "calls": [{"domain_id": "account_summary", "capability": "get_field", "args": {"key_path": "accountStatus"}}],
            "must_produce": [], "risk_if_missing": []
        },
        {
            "intent": "credit_limit",
            "calls": [{"domain_id": "account_summary", "capability": "get_field", "args": {"key_path": "creditLimit"}}],
            "must_produce": [], "risk_if_missing": []
        },
        {
            "intent": "due_date",
            "calls": [{"domain_id": "account_summary", "capability": "get_field", "args": {"key_path": "paymentDueDate"}}],
            "must_produce": [], "risk_if_missing": []
        },
        {
            "intent": "last_transaction",
            "calls": [{"domain_id": "transactions", "capability": "last_transaction", "args": {}}],
            "must_produce": [], "risk_if_missing": []
        },
        {
            "intent": "top_merchants_last_year",
            "calls": [{"domain_id": "transactions", "capability": "top_merchants", "args": {"period": "LAST_12M"}}],
            "must_produce": [], "risk_if_missing": []
        },
        {
            "intent": "why_interest",
            "calls": [
                {"domain_id": "statements", "capability": "total_interest",     "args": {"period": None, "nonzero": True}},
                {"domain_id": "statements", "capability": "interest_breakdown", "args": {"period": None, "nonzero": True}},
                {"domain_id": "statements", "capability": "trailing_interest",  "args": {"period": None}}
            ],
            "must_produce": [], "risk_if_missing": []
        },
    ]

def _examples_block() -> str:
    """Flatten EXAMPLES to a compact JSON-lines block so the LLM can learn patterns quickly."""
    return "\n".join(json.dumps(e, separators=(",", ":")) for e in EXAMPLES)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

import os

def llm_plan(question: str) -> Dict[str, Any]:
    """
    Build a plan by calling the LLM with:
      - context packs (planner mode)
      - routing hint (optional)
      - field map (so it can choose get_field)
      - system contract + lightweight examples
    Returns a dict with defaults if parsing fails.
    """
    cfg = _read_app_cfg()
    llm = build_llm_from_config(cfg)

    # Contextual system packs (planner mode)
    ctx = build_context(intent="planner", question=question, plan=None)

    # Optional routing hint for quick tool selection
    hint = build_hint_for_question(question)
    hint_msg = [{"role": "system", "content": hint}] if hint else []

    # Teach the model your canonical keys + synonyms (helps it pick get_field)
    field_help = {
        "domain": "account_summary",
        "canonical_keys": list(ACCOUNT_SUMMARY_FIELDS.keys()),
        "synonyms": ACCOUNT_SUMMARY_FIELDS,
    }
    field_help_msg = [{"role": "system", "content": f"Field map: {json.dumps(field_help)}"}]

    # Messages
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

    # Call LLM
    raw = llm.complete(messages, model=(cfg.get("llm") or {}).get("model"), temperature=0)

    # Extract the outermost JSON if the model wrapped it in text
    start = raw.find("{"); end = raw.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        raw = raw[start:end + 1]

    try:
        obj = json.loads(raw)
    except Exception:
        # Fallback default
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    # Normalize
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