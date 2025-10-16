# core/orchestrator/plan_llm.py
from __future__ import annotations
import os, json, yaml, httpx
from pathlib import Path
from typing import Any, Dict, List

# Optional context packs (safe fallbacks if missing)
try:
    from core.context.builder import build_context
except Exception:
    def build_context(intent: str, question: str, plan=None):
        return {"system": "You are the planner.", "context_msgs": []}

try:
    from core.context.hints import build_hint_for_question
except Exception:
    def build_hint_for_question(q: str) -> str:
        return ""

# Domain contract from the registry (so you don't hardcode capabilities)
try:
    from core.domains.registry import REGISTRY
    DOM_CONTRACT = REGISTRY.contract()
except Exception:
    DOM_CONTRACT = (
        "transactions:{get_field,list_where,sum_where,topk_by_sum,find_latest,semantic_search}; "
        "payments:{get_field,list_where}; "
        "statements:{get_field,list_where,sum_where,find_latest}; "
        "accounts:{get_field}"
    )

def _read_app_cfg(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists(): return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, messages, model=None, temperature=0):
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model or self.model, "messages": messages, "temperature": float(temperature)}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                # print server error for quick diagnosis
                print("[LLM ERROR]", r.status_code, r.text)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

def _coerce_chat_messages(msgs):
    out = []
    for m in msgs or []:
        if not isinstance(m, dict): continue
        role = m.get("role"); content = m.get("content")
        if content is None: continue
        if isinstance(content, (str, list)):
            out.append({"role": role, "content": content}); continue
        if isinstance(content, dict):
            inner = content.get("content") if "content" in content else None
            out.append({"role": role, "content": inner if isinstance(inner, str) else json.dumps(content, ensure_ascii=False)})
            continue
        out.append({"role": role, "content": str(content)})
    return out

def build_llm_from_config(cfg: Dict[str, Any]) -> _LLMClient:
    llm_cfg = (cfg.get("llm") or {})
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    if api_base and not api_base.startswith("http"): api_base = "https://" + api_base
    api_key = (llm_cfg.get("api_key") or os.getenv(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "") or llm_cfg.get("key") or "").strip()
    model   = (llm_cfg.get("model") or "gpt-4o-mini").strip()
    if not api_key:
        raise RuntimeError("Missing LLM API key. Set llm.api_key or export env named by llm.api_key_env.")
    return _LLMClient(api_base=api_base, api_key=api_key, model=model)

SYSTEM_CONTRACT = (
  "You are a planner for a banking copilot. "
  "Given a user question, output STRICT JSON: "
  "{intent:string, calls:[{domain_id, capability, args, strategy?, allow_rag_fallback?}], must_produce:[], risk_if_missing:[]}. "
  f"Domains/capabilities: {DOM_CONTRACT}. "
  "Set strategy:'deterministic' for direct, structured asks (exact fields, amounts, dates). "
  "Set strategy:'rag' for WHY/HOW explanations, policy/handbook queries, or ambiguous asks needing multi-doc evidence. "
  "Use strategy:'auto' if unsure; default allow_rag_fallback:true. "
  "Prefer transactions to include only POSTED by default unless user asks for pending. "
  "Return ONLY JSON."
)

def make_plan(question: str, app_yaml_path: str) -> Dict[str, Any]:
    cfg = _read_app_cfg(app_yaml_path)
    llm = build_llm_from_config(cfg)

    ctx = build_context(intent="planner", question=question, plan=None)
    hint = build_hint_for_question(question)
    hint_msg = [{"role": "system", "content": hint}] if hint else []

    messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": ctx.get("system", "You are the planner.")}]
        + ctx.get("context_msgs", [])
        + hint_msg
        + [{"role": "system", "content": SYSTEM_CONTRACT}]
        + [{"role": "user", "content": f"Return ONLY JSON. Question: {question}"}]
    )
    messages = _coerce_chat_messages(messages)
    raw = llm.complete(messages, model=(cfg.get("llm") or {}).get("model"), temperature=0)

    # Strip to outermost JSON
    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s: raw = raw[s:e+1]
    try:
        obj = json.loads(raw)
    except Exception:
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    # sanitize
    if not isinstance(obj, dict): obj = {}
    obj.setdefault("intent", "unknown")
    safe_calls = []
    for c in obj.get("calls", []) or []:
        if not isinstance(c, dict): continue
        c.setdefault("domain_id", ""); c.setdefault("capability", ""); c.setdefault("args", {})
        if not isinstance(c["args"], dict): c["args"] = {}
        c["strategy"] = (c.get("strategy") or "auto").lower()
        c["allow_rag_fallback"] = bool(c.get("allow_rag_fallback", True))
        safe_calls.append(c)
    obj["calls"] = safe_calls
    obj.setdefault("must_produce", [])
    obj.setdefault("risk_if_missing", [])
    return obj