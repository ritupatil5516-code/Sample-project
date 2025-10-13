# core/orchestrator/plan_llm.py
from __future__ import annotations
import json, os
from typing import Any, Dict, List
from pathlib import Path
import httpx
import yaml

# ---- optional context packs (fallback to no-ops if missing) -----------------
try:
    from core.context.builder import build_context
except Exception:
    def build_context(intent: str, question: str, plan: Dict[str, Any] | None) -> Dict[str, Any]:
        return {"system": "You are a planner for a credit-card copilot.", "context_msgs": []}

try:
    from core.context.hints import build_hint_for_question
except Exception:
    def build_hint_for_question(question: str) -> str | None:
        return None

CONFIG_FILE = "config/app.yaml"

# ------------------------------ config helpers --------------------------------
def _read_config_files(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _to_text(resp: httpx.Response | Dict[str, Any] | str) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        return json.dumps(resp, ensure_ascii=False)
    try:
        return resp.text
    except Exception:
        return str(resp)

# ------------------------------- LLM wrapper ----------------------------------
class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> Dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        with httpx.Client(timeout=45.0) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                print("[LLM ERROR]", r.status_code, r.text)
            r.raise_for_status()
            return r.json()

def build_llm(cfg: Dict[str, Any]) -> _LLMClient:
    llm_cfg = (cfg.get("llm") or {})
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    api_key  = (llm_cfg.get("api_key") or
                os.getenv(llm_cfg.get("api_key_env", "") or "OPENAI_API_KEY", "") or
                llm_cfg.get("key") or "").strip()
    model    = (llm_cfg.get("model") or "gpt-4o-mini").strip()
    if api_base and not api_base.startswith(("http://", "https://")):
        api_base = "https://" + api_base
    if not api_key:
        raise RuntimeError("Missing LLM API key. Configure llm.api_key or llm.api_key_env.")
    return _LLMClient(api_base=api_base, api_key=api_key, model=model)

# ------------------------------ Planning prompt -------------------------------
SYSTEM_CONTRACT = (
  "You are a planner for a credit-card copilot. "
  "Given a user question, output STRICT JSON ONLY:\n"
  "{intent: string, calls: [{domain_id, capability, args}], must_produce:[], risk_if_missing:[]}.\n"
  "Domains/capabilities:\n"
  "transactions:{last_transaction, top_merchants, average_per_month, spend_in_period, "
  "              list_over_threshold, purchases_in_cycle, semantic_search, find_by_merchant};\n"
  "payments:{last_payment, total_credited_year, payments_in_period};\n"
  "statements:{total_interest, interest_breakdown, trailing_interest};\n"
  "account_summary:{current_balance, available_credit, get_field};\n"
  "rag:{unified_answer, account_answer, knowledge_answer}.\n"
  "Also support a generic 5-op DSL for any domain when appropriate:\n"
  "ops:{get_field|find_latest|sum_where|topk_by_sum|list_where|semantic_search} with args object.\n"
  "Routing guidance:\n"
  "- Use exact field/calculator when the user asks for a concrete value (e.g., current balance, available credit).\n"
  "- Use rag.unified_answer for broad/why/how questions needing JSON + handbook/policy.\n"
  "- Use rag.account_answer when only account JSONs are needed; rag.knowledge_answer for handbook/policy only.\n"
  "- If user asks for last/latest interest or omits period, set args.period=null; if they say 'last interest' also set args.nonzero=true.\n"
  "Return ONLY JSON."
)

EXAMPLES: List[Dict[str, Any]] = [
    {
        "intent": "get_current_balance",
        "calls": [
            {"domain_id": "account_summary", "capability": "current_balance", "args": {"account_id": "AID"}}
        ],
        "must_produce": [], "risk_if_missing": []
    },
    {
        "intent": "where_spent_most",
        "calls": [
            {"domain_id": "transactions", "capability": "top_merchants",
             "args": {"account_id": "AID", "period": "LAST_12M"}}
        ],
        "must_produce": [], "risk_if_missing": []
    },
    {
        "intent": "why_fee",
        "calls": [
            {"domain_id": "rag", "capability": "unified_answer",
             "args": {"account_id": "AID", "k": 6}}
        ],
        "must_produce": [], "risk_if_missing": []
    },
    {
        "intent": "policy_question",
        "calls": [
            {"domain_id": "rag", "capability": "knowledge_answer", "args": {"k": 5}}
        ],
        "must_produce": [], "risk_if_missing": []
    }
]

def _examples_block() -> str:
    return "\n".join(json.dumps(e, separators=(",", ":")) for e in EXAMPLES)

def _coerce_chat_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, (str, list)):
            out.append({"role": role, "content": content})
        elif isinstance(content, dict) and isinstance(content.get("content"), str):
            out.append({"role": role, "content": content["content"]})
        else:
            out.append({"role": role, "content": json.dumps(content, ensure_ascii=False)})
    return out

def build_chat_messages(ctx: Dict[str, Any], question: str, hint: str | None) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": ctx.get("system", "")}]
    msgs += ctx.get("context_msgs", [])
    if hint:
        msgs.append({"role": "system", "content": hint})
    msgs.append({"role": "system", "content": SYSTEM_CONTRACT})
    msgs.append({
        "role": "user",
        "content": (
            "Return ONLY JSON. Do not include explanations or code fences.\n\n"
            f"Examples:\n{_examples_block()}\n\n"
            f"Question: {question}"
        )
    })
    return _coerce_chat_messages(msgs)

# --------------------------------- Public API ---------------------------------
def llm_plan(question: str) -> Dict[str, Any]:
    """
    Build a plan by calling the LLM with context packs + routing hints + contract.
    Returns a dict with defaults if parsing fails.
    """
    cfg = _read_config_files(CONFIG_FILE)
    Settings = type("Settings", (), {})  # tiny holder
    Settings.llm = build_llm(cfg)

    ctx  = build_context(intent="planner", question=question, plan=None)
    hint = build_hint_for_question(question)
    chat_messages = build_chat_messages(ctx, question, hint)

    resp = Settings.llm.chat(chat_messages, temperature=0.0)
    raw_text = _to_text(resp)

    # Strip to outermost JSON if wrapped
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        raw_text = raw_text[start:end+1]

    try:
        obj = json.loads(raw_text)
    except Exception:
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    if not isinstance(obj, dict):
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    # sanitize calls/args
    obj.setdefault("intent", "unknown")
    safe_calls: List[Dict[str, Any]] = []
    for c in obj.get("calls") or []:
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