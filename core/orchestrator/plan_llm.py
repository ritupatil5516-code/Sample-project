# core/orchestrator/plan_llm.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Your existing helpers (keep these imports the same as in your project)
from core.context.builder import build_context
from core.context.hints import build_hint_for_question

# Runtime LLM factory (set up during FastAPI startup)
# Must expose RUNTIME.chat() -> an LLM client that supports .invoke(messages) or returns an object with .content
from src.core.runtime import RUNTIME  # adjust import if your runtime module lives elsewhere


# ------------------------------- file loaders ---------------------------------

_PACK_PATH = Path("core/context/packs/core.yaml")


def _read_pack() -> Dict[str, Any]:
    """
    Load the core planning pack (system text, glossary, reasoning, planner rules, contract).
    """
    if not _PACK_PATH.exists():
        return {}
    try:
        return yaml.safe_load(_PACK_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _extract_planner_rules(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Return just the planner_rules block (synonyms, defaults, routes)."""
    pr = (pack.get("planner_rules") or {}) if isinstance(pack, dict) else {}
    pr.setdefault("synonyms", {})
    pr.setdefault("defaults", {})
    pr.setdefault("routes", [])
    return pr


# ------------------------------ tiny text utils -------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall(_norm(s))


def _contains_any(tokens: List[str], bag: List[str]) -> bool:
    st = set(tokens)
    return any(w in st for w in bag)


def _expand_syn_bucket(bucket: Any) -> List[str]:
    """
    bucket may be ["recent","latest"] or {"recent":["latest","most recent"]}.
    We always return a flat list of keywords in lowercase.
    """
    if isinstance(bucket, list):
        return [_norm(x) for x in bucket if x]
    if isinstance(bucket, dict):
        out: List[str] = []
        for v in bucket.values():
            out.extend(_expand_syn_bucket(v))
        return out
    if isinstance(bucket, str):
        return [_norm(bucket)]
    return []


def _resolve_synonyms(tokens: List[str], synonyms: Dict[str, Any]) -> Dict[str, bool]:
    """
    Map each synonym bucket name -> True/False if present in the question.
    """
    flags: Dict[str, bool] = {}
    for name, bucket in (synonyms or {}).items():
        bag = _expand_syn_bucket(bucket)
        flags[name] = _contains_any(tokens, bag)
    return flags


# ------------------------------ arg templating --------------------------------

_TMPL_RE = re.compile(r"\$\{([A-Z0-9_\-\*]+)\}")

def _infer_date_defaults(q: str, defaults: Dict[str, Any]) -> Dict[str, str]:
    """
    Very small heuristics for date placeholders:
      ${YYYY-MM}, ${YYYY-Q*}, LAST_MONTH, PREV_MONTH, THIS_YEAR, LAST_12M
    You can expand this as needed.
    """
    qn = _norm(q)
    out: Dict[str, str] = {}
    # YYYY-MM in text?
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", qn)
    if m:
        out["YYYY-MM"] = f"{m.group(1)}-{m.group(2)}"
    # quarter like 2025 q3
    mq = re.search(r"(20\d{2})\s*[- ]?q([1-4])", qn)
    if mq:
        out["YYYY-Q*"] = f"{mq.group(1)}-Q{mq.group(2)}"
    return out


def _fill_templates(obj: Any, question: str, defaults: Dict[str, Any]) -> Any:
    """
    Recursively replace ${TOKENS} in args with inferred values or defaults.
    """
    date_inf = _infer_date_defaults(question, defaults)

    def _subst(s: str) -> str:
        def repl(m):
            key = m.group(1)
            # Prefer inferred
            if key in date_inf:
                return date_inf[key]
            # Fallback default-> exact token
            # e.g., no_period_for_spend -> LAST_12M, but we only see literal tokens like LAST_12M in routes
            # so if token itself is a named default, use that value
            return defaults.get(key, key)
        return _TMPL_RE.sub(repl, s)

    if isinstance(obj, str):
        return _subst(obj)
    if isinstance(obj, list):
        return [_fill_templates(x, question, defaults) for x in obj]
    if isinstance(obj, dict):
        return {k: _fill_templates(v, question, defaults) for k, v in obj.items()}
    return obj


# ------------------------------ rule-based plan -------------------------------

def _must_tokens_present(
    q_tokens: List[str],
    syn_flags: Dict[str, bool],
    must: List[str],
) -> bool:
    """
    'must' contains literal words OR synonym-bucket names.
    If a token exists in synonyms dict, we check its flag; else check literal in q_tokens.
    """
    st = set(q_tokens)
    for m in must or []:
        m = _norm(m)
        if m in syn_flags:
            if not syn_flags[m]:
                return False
        else:
            if m and (m not in st):
                return False
    return True


def _try_plan_from_rules(question: str, rules_pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tokens = _tokens(question)
    synonyms = rules_pack.get("synonyms") or {}
    defaults = rules_pack.get("defaults") or {}
    routes = rules_pack.get("routes") or []

    syn_flags = _resolve_synonyms(tokens, synonyms)

    for r in routes:
        must = r.get("must") or []
        if not _must_tokens_present(tokens, syn_flags, must):
            continue

        call = r.get("call") or {}
        # Deep copy + template fill
        call = json.loads(json.dumps(call))
        call["args"] = _fill_templates(call.get("args") or {}, question, defaults)

        plan: Dict[str, Any] = {
            "intent": r.get("name") or "unknown",
            "calls": [call],
            "must_produce": [],
            "risk_if_missing": [],
        }
        # pass through strategy if present (e.g., "rag:unified")
        if r.get("strategy"):
            plan["strategy"] = str(r["strategy"])
            # also copy onto the call so the executor can branch per-call if needed
            plan["calls"][0]["strategy"] = str(r["strategy"])
        return plan

    return None


# ------------------------------ LLM fallback ----------------------------------

def _build_chat_messages(ctx: Dict[str, Any], pack: Dict[str, Any], question: str, rules_pack: Dict[str, Any], hint: Optional[str]) -> List[Dict[str, Any]]:
    """
    Build a conservative prompt for the planner LLM.
    We include:
      - system pack text (system + glossary + reasoning)
      - planner contract from pack
      - a few inline examples if you keep them elsewhere (optional)
    """
    sys_parts: List[str] = []
    if pack.get("system"):
        sys_parts.append(str(pack["system"]).strip())
    if pack.get("glossary"):
        gl = pack["glossary"]
        if isinstance(gl, list):
            sys_parts.append("Glossary:\n- " + "\n- ".join(map(str, gl)))
    if pack.get("reasoning"):
        rs = pack["reasoning"]
        if isinstance(rs, list):
            sys_parts.append("Reasoning rules:\n- " + "\n- ".join(map(str, rs)))

    planner_contract = (pack.get("planner_contract") or
                        "Return a strict JSON plan with {intent, calls:[{domain_id, capability, args}], must_produce:[], risk_if_missing:[]}. NO prose.")
    sys_text = "\n\n".join(sys_parts + [str(planner_contract).strip()])

    messages: List[Dict[str, Any]] = []
    # context system (if your builder emits it)
    if ctx and ctx.get("system"):
        messages.append({"role": "system", "content": ctx["system"]})
    # additional context msgs from your builder (optional)
    for m in (ctx.get("context_msgs") or []):
        if isinstance(m, dict) and m.get("role") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})

    # core pack system + contract
    messages.append({"role": "system", "content": sys_text})

    # add hint if any
    if hint:
        messages.append({"role": "system", "content": str(hint)})

    # final user turn
    messages.append({
        "role": "user",
        "content": (
            "Return ONLY JSON. Do not include explanations or markdown code fences.\n\n"
            f"Question: {question}"
        )
    })
    return messages


def _coerce_response_text(resp: Any) -> str:
    """
    Accept:
      - string
      - dict with 'content'
      - LangChain BaseMessage with .content
      - OpenAI httpx raw dict {choices[0].message.content}
    Return a plain string.
    """
    if resp is None:
        return ""
    # LangChain / pydantic messages
    if hasattr(resp, "content") and isinstance(getattr(resp, "content"), str):
        return resp.content
    # openai raw dict
    if isinstance(resp, dict):
        # openai/chat-completions shape
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            pass
        if "content" in resp and isinstance(resp["content"], str):
            return resp["content"]
    # string
    if isinstance(resp, str):
        return resp
    return str(resp)


def _extract_json_block(s: str) -> str:
    """Grab outermost {...} from possibly noisy LLM text."""
    if not s:
        return ""
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        return s[start: end + 1]
    return ""


# --------------------------------- Public API ---------------------------------

def llm_plan(question: str) -> Dict[str, Any]:
    """
    Try rules first (from YAML pack). If nothing matches, fall back to the runtime LLM.
    Always returns a dict with {intent, calls, must_produce, risk_if_missing}.
    """
    # Load pack + rules
    pack = _read_pack()
    rules = _extract_planner_rules(pack)

    # 1) Rule-first planner
    rule_plan = _try_plan_from_rules(question, rules)
    if rule_plan:
        return rule_plan

    # 2) LLM fallback with your context builder / hinting
    ctx = build_context(intent="planner", question=question, plan=None)
    hint = build_hint_for_question(question)
    messages = _build_chat_messages(ctx, pack, question, rules, hint)

    # Call the runtime LLM (must be set up in runtime.startup)
    llm = RUNTIME.chat()
    # Support both: LangChain ChatOpenAI (expects list[dict] OK via LC’s conversion) or your own client.
    try:
        resp = llm.invoke(messages)  # LangChain style
    except AttributeError:
        # If it's your custom client with .complete(messages, temperature=0)
        resp = getattr(llm, "complete")(messages, temperature=0)

    raw_text = _coerce_response_text(resp)
    json_block = _extract_json_block(raw_text)

    try:
        obj = json.loads(json_block)
    except Exception:
        # Safe default
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    # Normalize
    if not isinstance(obj, dict):
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    obj.setdefault("intent", "unknown")
    calls_in = obj.get("calls") or []
    calls_out: List[Dict[str, Any]] = []
    for c in calls_in:
        if not isinstance(c, dict):
            continue
        # normalize fields
        dom = str(c.get("domain_id", "")).strip()
        cap = str(c.get("capability", "")).strip()
        args = c.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        call_norm = {"domain_id": dom, "capability": cap, "args": args}
        # if planner added "strategy", keep it (executor can branch deterministic vs rag)
        if "strategy" in c:
            call_norm["strategy"] = str(c["strategy"])
        calls_out.append(call_norm)

    obj["calls"] = calls_out
    obj.setdefault("must_produce", [])
    obj.setdefault("risk_if_missing", [])

    return obj