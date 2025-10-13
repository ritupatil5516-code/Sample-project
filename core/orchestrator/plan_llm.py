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

from core.orchestrator.intent_examples import INTENT_EXAMPLES

cfg = yaml.safe_load(open("config/app.yaml").read())
key_env = (cfg.get("llm", {}).get("api_key_env") or "OPENAI_API_KEY").strip()
key = (os.getenv(key_env) or "").strip()
base = (cfg.get("llm", {}).get("base_url") or "https://api.openai.com/v1").strip()

print("[PLANNER DEBUG] base_url =", base)
print("[PLANNER DEBUG] key_env =", key_env, "present?" , bool(key))
# ------------------------------ config helpers --------------------------------

def _coerce_chat_messages(msgs):
    """
    Ensure every message has str content (or a valid OpenAI content array).
    - dict content with 'content' str -> keep
    - dict content without 'content' -> json.dumps
    - None -> drop the message
    - list content (already an array-of-parts) -> keep as-is
    """
    out = []
    for i, m in enumerate(msgs or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if content is None:
            # drop empty messages
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            # Assume it's already OpenAI "array of parts" format
            out.append({"role": role, "content": content})
            continue
        if isinstance(content, dict):
            # If nested dict with 'content' str, unwrap; else JSON-serialize
            inner = content.get("content") if "content" in content else None
            if isinstance(inner, str):
                out.append({"role": role, "content": inner})
            else:
                out.append({"role": role, "content": json.dumps(content, ensure_ascii=False)})
            continue
        # Fallback: stringify anything else
        out.append({"role": role, "content": str(content)})
    return out

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
    def __init__(self, api_base, api_key, model):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, messages, model=None, temperature=0):
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
                # If failure, print server response to see *why*
                if r.status_code >= 400:
                    print("[LLM ERROR]", r.status_code, r.text)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            # Surface the API error text to Streamlit to diagnose quickly
            raise RuntimeError(f"OpenAI 400: {e.response.text}") from e



def build_llm_from_config(cfg: Dict[str, Any]) -> _LLMClient:
    import os
    llm_cfg = (cfg.get("llm") or {})

    api_base = llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1"
    # Prefer explicit api_key, else read from env specified by api_key_env
    api_key = (llm_cfg.get("api_key") or
               os.getenv(llm_cfg.get("api_key_env", ""), "") or
               llm_cfg.get("key") or "")
    model   = llm_cfg.get("model") or "gpt-4o-mini"

    api_base = api_base.strip()
    if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
        api_base = "https://" + api_base

    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing LLM API key. Set llm.api_key in config/app.yaml or export the env var named by llm.api_key_env."
        )

    return _LLMClient(api_base=api_base, api_key=api_key, model=model)
# ------------------------------ Planning prompt -------------------------------

SYSTEM_CONTRACT = (
  "You are the planner for a credit-card copilot.\n"
  "Output STRICT JSON with this schema:\n"
  "{\n"
  "  intent: string,\n"
  "  ops: [\n"
  "    { op: 'get_field'|'find_latest'|'sum_where'|'topk_by_sum'|'list_where'|'semantic_search',\n"
  "      domain: 'account_summary'|'statements'|'payments'|'transactions',\n"
  "      args: object }\n"
  "  ],\n"
  "  must_produce:[],\n"
  "  risk_if_missing:[]\n"
  "}\n"
  "Guidance:\n"
  "- Direct fields (e.g., account status, available credit) -> get_field(domain, key_path).\n"
  "- Latest questions (e.g., last statement closing date, latest interest charged) -> find_latest(domain, ts_field, value_path[, where]).\n"
  "- Totals (e.g., total posted purchase spend) -> sum_where(domain, value_path, where).\n"
  "- Top-K (e.g., top merchants by spend) -> topk_by_sum(domain, group_key, value_path, where, k).\n"
  "- Listings (e.g., list posted purchases over $X, find purchases at <merchant>) -> list_where(domain, where, sort_by, desc, limit).\n"
  "- Fuzzy concept search (e.g., 'travel purchases') -> semantic_search(domain, query, k).\n"
  "Use these schema hints:\n"
  "account_summary: accountStatus, highestPriorityStatus, flags[], subStatuses[], currentBalance, availableCreditAmount, minimumDueAmount, paymentDueAmount, openedDate, closedDate.\n"
  "statements: closingDateTime, openingDateTime, dueDateTime, dueDate, interestCharged, feesCharged, period.\n"
  "payments: paymentDateTime, paymentPostedDateTime, amount.\n"
  "transactions: transactionDateTime, postedDateTime, amount, merchantName, transactionType, displayTransactionType, transactionStatus.\n"
  "Return ONLY JSON."
)

# A few tiny patterns so the model learns tool selection quickly.
EXAMPLES = INTENT_EXAMPLES

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

    messages = _coerce_chat_messages(messages)

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