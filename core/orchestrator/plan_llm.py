from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

from core.context.builder import build_context
from core.context.hints import build_hint_for_question

CORE_PACK = Path("core/context/packs/core.yaml")
APP_CFG   = Path("config/app.yaml")

# ------------------------------ tiny LLM client ------------------------------ #

class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, messages: List[Dict[str, Any]], model: Optional[str] = None, temperature: float = 0.0) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model or self.model, "messages": messages, "temperature": float(temperature)}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"LLM {r.status_code}: {r.text}")
            data = r.json()
            return data["choices"][0]["message"]["content"]

# ------------------------------ config helpers ------------------------------ #

def _read_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _read_app_cfg() -> Dict[str, Any]:
    return _read_yaml(APP_CFG)

def _read_core_pack() -> Dict[str, Any]:
    return _read_yaml(CORE_PACK)

def build_llm_from_config(cfg: Dict[str, Any]) -> _LLMClient:
    llm_cfg = (cfg.get("llm") or {})
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    model    = (llm_cfg.get("model") or "gpt-4o-mini").strip()
    key = (llm_cfg.get("api_key")
           or (llm_cfg.get("api_key_env") and __import__("os").getenv(llm_cfg.get("api_key_env")))
           or llm_cfg.get("key") or "")
    key = (key or "").strip()
    if not key:
        raise RuntimeError("Missing LLM API key (set llm.api_key or export llm.api_key_env).")
    if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
        api_base = "https://" + api_base
    return _LLMClient(api_base=api_base, api_key=key, model=model)

# ------------------------------ examples (small) ----------------------------- #

EXAMPLES: List[Dict[str, Any]] = [
    {
        "intent": "get_account_status",
        "calls": [{"domain_id": "account_summary", "capability": "get_field", "args": {"field": "status"}}],
        "must_produce": [], "risk_if_missing": [], "strategy": "deterministic"
    },
    {
        "intent": "last_interest_charge",
        "calls": [{"domain_id": "statements", "capability": "find_latest",
                   "args": {"field": "closingDateTime", "where": {"interestCharged": {">": 0}}}}],
        "must_produce": [], "risk_if_missing": [], "strategy": "deterministic"
    },
]

def _examples_block() -> str:
    return "\n".join(json.dumps(e, separators=(",", ":")) for e in EXAMPLES)

def _coerce_chat_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs or []:
        if not isinstance(m, dict): continue
        role = m.get("role")
        content = m.get("content")
        if content is None: continue
        if isinstance(content, (str, list)):
            out.append({"role": role, "content": content}); continue
        if isinstance(content, dict):
            inner = content.get("content") if "content" in content else None
            out.append({"role": role, "content": inner if isinstance(inner, str) else json.dumps(content, ensure_ascii=False)})
            continue
        out.append({"role": role, "content": str(content)})
    return out

# ----------------------- rules sourced from core.yaml ------------------------ #

def _extract_planner_rules(core_pack: Dict[str, Any]) -> Dict[str, Any]:
    pr = core_pack.get("planner_rules") or {}
    pr.setdefault("synonyms", {}); pr.setdefault("routes", [])
    return pr

def _token_present(q: str, tokens: List[str]) -> bool:
    ql = " " + (q or "").lower() + " "
    for raw in tokens or []:
        t = (raw or "").strip().lower()
        if not t: continue
        if f" {t} " in ql or t in ql:
            return True
    return False

def try_plan_from_rules(question: str, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    syn = rules.get("synonyms") or {}
    for r in (rules.get("routes") or []):
        must = r.get("must") or []
        ok = True
        for tag in must:
            tokens = syn.get(tag) if tag in syn else [tag]
            if not _token_present(question, tokens):
                ok = False; break
        if ok:
            call = r.get("call") or {}
            return {
                "intent": r.get("name", "rule_match"),
                "calls": [{
                    "domain_id": call.get("domain_id", ""),
                    "capability": call.get("capability", ""),
                    "args": call.get("args", {}) or {}
                }],
                "must_produce": [],
                "risk_if_missing": [],
                "strategy": "deterministic"
            }
    return None

# -------------------------- message construction ----------------------------- #

def build_chat_messages(ctx: Dict[str, Any], core_pack: Dict[str, Any], question: str, rules: Dict[str, Any], hint: Optional[str]) -> List[Dict[str, Any]]:
    sys_txt = (core_pack.get("system") or ctx.get("system") or "").strip()
    glossary = core_pack.get("glossary") or []
    reasoning = core_pack.get("reasoning") or []
    planner_contract = (core_pack.get("planner_contract") or
                        "Return ONLY JSON with {intent, calls, must_produce, risk_if_missing}. Prefer DSL ops.")
    hint_msg = [{"role": "system", "content": hint}] if hint else []

    extra_msgs: List[Dict[str, str]] = []
    if glossary:
        extra_msgs.append({"role": "system", "content": "Glossary:\n- " + "\n- ".join(map(str, glossary))})
    if reasoning:
        extra_msgs.append({"role": "system", "content": "Reasoning guide:\n- " + "\n- ".join(map(str, reasoning))})

    rules_yaml = yaml.safe_dump(rules, allow_unicode=True)
    rules_msg = {"role": "system", "content": "Planner routing rules:\n\n" + rules_yaml}

    messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": sys_txt}]
        + ctx.get("context_msgs", [])
        + extra_msgs
        + hint_msg
        + [rules_msg]
        + [{"role": "system", "content": planner_contract}]
        + [{
            "role": "user",
            "content": (
                "Return ONLY JSON. Do not include explanations or markdown code fences.\n\n"
                f"Examples:\n{_examples_block()}\n\n"
                f"Question: {question}"
            )
        }]
    )
    return _coerce_chat_messages(messages)

# --------------------------------- public API -------------------------------- #

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from src.core.runtime import RUNTIME

def _to_lc_messages(msgs: list[dict]) -> list:
    out = []
    for m in msgs:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out

def _extract_json_block(text: str) -> str:
    s = str(text or "")
    # strip code fences if present
    if "```" in s:
        s = s.split("```", 1)[-1]
        if "```" in s:
            s = s.split("```", 1)[0]
    start = s.find("{"); end = s.rfind("}")
    return s[start:end+1] if (start >= 0 and end > start) else s

def llm_plan(question: str) -> dict:
    # ... build context/hint and your `messages` (list of dicts: {role, content}) ...
    ctx   = build_context(intent="planner", question=question, plan=None)
    hint  = build_hint_for_question(question)
    messages = build_chat_messages(ctx, core_pack=_read_pack(), question=question, rules=_extract_planner_rules(_read_pack()), hint=hint)

    # LangChain call (IMPORTANT: use .content)
    chat = RUNTIME.chat()  # returns langchain_openai.ChatOpenAI
    lc_messages = _to_lc_messages(messages)
    ai_msg = chat.invoke(lc_messages)            # AIMessage
    raw_text = getattr(ai_msg, "content", str(ai_msg))  # <-- string for parsing

    # Parse JSON result
    try:
        json_block = _extract_json_block(raw_text)
        obj = json.loads(json_block)
    except Exception:
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    # normalize
    if not isinstance(obj, dict):
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}
    obj.setdefault("intent", "unknown")
    calls = obj.get("calls") or []
    safe = []
    for c in calls:
        if isinstance(c, dict):
            c.setdefault("domain_id", ""); c.setdefault("capability", ""); c.setdefault("args", {})
            if not isinstance(c["args"], dict): c["args"] = {}
            safe.append(c)
    obj["calls"] = safe
    obj.setdefault("must_produce", []); obj.setdefault("risk_if_missing", [])
    return obj