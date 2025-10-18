# core/orchestrator/plan_llm.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Context builders (keep your originals)
from core.context.builder import build_context
from core.context.hints import build_hint_for_question

# Runtime LLM (set in FastAPI startup)
from src.core.runtime import RUNTIME  # expose .chat() LangChain ChatOpenAI (or equivalent)

# -----------------------------------------------------------------------------#
# Limits & aliases
# -----------------------------------------------------------------------------#

ALLOWED_DOMAINS = {"transactions", "payments", "statements", "accounts"}
DOMAIN_ALIASES = {
    "account_summary": "accounts",
    "account": "accounts",
    "summary": "accounts",
}

PACK_PATH = Path("core/context/packs/core.yaml")

WORD_RE = re.compile(r"[a-z0-9]+", re.I)


# -----------------------------------------------------------------------------#
# Pack / rules loading
# -----------------------------------------------------------------------------#

def _read_pack() -> Dict[str, Any]:
    if not PACK_PATH.exists():
        return {}
    try:
        return yaml.safe_load(PACK_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _planner_rules(pack: Dict[str, Any]) -> Dict[str, Any]:
    pr = (pack.get("planner_rules") or {}) if isinstance(pack, dict) else {}
    pr.setdefault("synonyms", {})
    pr.setdefault("defaults", {})
    pr.setdefault("routes", [])
    return pr


# -----------------------------------------------------------------------------#
# Text helpers
# -----------------------------------------------------------------------------#

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _tokens(s: str) -> List[str]:
    return WORD_RE.findall(_norm(s))


def _expand_syn_bucket(bucket: Any) -> List[str]:
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
    flags: Dict[str, bool] = {}
    st = set(tokens)
    for name, bucket in (synonyms or {}).items():
        bag = set(_expand_syn_bucket(bucket))
        flags[name] = any(b in st for b in bag)
    return flags


def _must_ok(q_tokens: List[str], syn_flags: Dict[str, bool], must: List[str]) -> bool:
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


# -----------------------------------------------------------------------------#
# Templating minimal (YYYY-MM, YYYY-Q*)
# -----------------------------------------------------------------------------#

_TMPL_RE = re.compile(r"\$\{([A-Z0-9_\-\*]+)\}")

def _infer_dates(question: str) -> Dict[str, str]:
    q = _norm(question)
    out: Dict[str, str] = {}
    m = re.search(r"(20\d{2})[-/](0[1-9]|1[0-2])", q)
    if m:
        out["YYYY-MM"] = f"{m.group(1)}-{m.group(2)}"
    mq = re.search(r"(20\d{2})\s*[- ]?q([1-4])", q)
    if mq:
        out["YYYY-Q*"] = f"{mq.group(1)}-Q{mq.group(2)}"
    return out


def _fill_templates(obj: Any, question: str, defaults: Dict[str, Any]) -> Any:
    inferred = _infer_dates(question)

    def sub_once(s: str) -> str:
        def repl(m):
            key = m.group(1)
            if key in inferred:
                return inferred[key]
            return str(defaults.get(key, key))
        return _TMPL_RE.sub(repl, s)

    if isinstance(obj, str):
        return sub_once(obj)
    if isinstance(obj, list):
        return [_fill_templates(x, question, defaults) for x in obj]
    if isinstance(obj, dict):
        return {k: _fill_templates(v, question, defaults) for k, v in obj.items()}
    return obj


# -----------------------------------------------------------------------------#
# Strategy selection
# -----------------------------------------------------------------------------#

EXPLAIN_WORDS = {"why", "explain", "because", "reason"}
POLICY_WORDS = {"policy", "handbook", "agreement", "fee", "fees", "interest", "trailing"}

def _choose_strategy(question: str, explicit: Optional[str]) -> str:
    """
    Returns one of: 'deterministic', 'rag:unified', 'rag:knowledge'
    - If route provides explicit strategy -> honor it.
    - Else, if question looks explanatory/policy-like -> rag:unified
    - Otherwise -> deterministic
    """
    if explicit:
        return explicit
    toks = set(_tokens(question))
    if (toks & EXPLAIN_WORDS) or (toks & POLICY_WORDS):
        # default to unified so we can blend account JSON + policy
        return "rag:unified"
    return "deterministic"


# -----------------------------------------------------------------------------#
# Rule-first planner
# -----------------------------------------------------------------------------#

def _normalize_domain(dom: str) -> str:
    d = _norm(dom)
    d = DOMAIN_ALIASES.get(d, d)
    # clamp to allowed; default to 'accounts' if unknown
    return d if d in ALLOWED_DOMAINS else "accounts"


def _plan_from_rules(question: str, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tokens = _tokens(question)
    syn = rules.get("synonyms") or {}
    defaults = rules.get("defaults") or {}
    routes = rules.get("routes") or []

    syn_flags = _resolve_synonyms(tokens, syn)

    for r in routes:
        must = r.get("must") or []
        if not _must_ok(tokens, syn_flags, must):
            continue

        call = json.loads(json.dumps(r.get("call") or {}))  # deep copy
        call["args"] = _fill_templates(call.get("args") or {}, question, defaults)

        # normalize domain strictly
        dom = _normalize_domain(call.get("domain_id", ""))
        call["domain_id"] = dom

        # ensure strategy
        call_strategy = _choose_strategy(question, r.get("strategy"))
        call["strategy"] = call_strategy

        plan = {
            "intent": r.get("name") or "unknown",
            "calls": [call],
            "must_produce": [],
            "risk_if_missing": [],
            "strategy": call_strategy,  # mirror at plan level for convenience
        }
        return plan

    return None


# -----------------------------------------------------------------------------#
# LLM fallback (constrained)
# -----------------------------------------------------------------------------#

def _build_llm_messages(ctx: Dict[str, Any], pack: Dict[str, Any], question: str, hint: Optional[str]) -> List[Dict[str, Any]]:
    system_lines: List[str] = []

    # Core system / glossary / reasoning from pack
    if pack.get("system"):
        system_lines.append(str(pack["system"]).strip())

    gl = pack.get("glossary")
    if isinstance(gl, list) and gl:
        system_lines.append("Glossary:\n- " + "\n- ".join(map(str, gl)))

    rs = pack.get("reasoning")
    if isinstance(rs, list) and rs:
        system_lines.append("Reasoning rules:\n- " + "\n- ".join(map(str, rs)))

    # Constrained planner contract
    contract = (
        "You must return ONLY a strict JSON object:\n"
        "{\n"
        '  "intent": "string",\n'
        '  "calls": [\n'
        '    {"domain_id":"transactions|payments|statements|accounts", "capability":"get_field|find_latest|sum_where|topk_by_sum|list_where|semantic_search|compare_periods", "args":{...}, "strategy":"deterministic|rag:unified|rag:knowledge"}\n'
        "  ],\n"
        '  "must_produce": [],\n'
        '  "risk_if_missing": [],\n'
        '  "strategy": "deterministic|rag:unified|rag:knowledge"\n'
        "}\n"
        "- Domains allowed: transactions, payments, statements, accounts (no others!)\n"
        "- If the question asks WHY/EXPLAIN or cites policy/handbook/fees/interest policy, set strategy to rag:unified or rag:knowledge.\n"
        "- Otherwise set strategy to deterministic.\n"
        "- Do NOT include markdown fences or commentary."
    )
    system_lines.append(contract)

    messages: List[Dict[str, Any]] = []
    if ctx and ctx.get("system"):
        messages.append({"role": "system", "content": ctx["system"]})
    for m in (ctx.get("context_msgs") or []):
        if isinstance(m, dict) and m.get("role") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "system", "content": "\n\n".join(system_lines)})

    if hint:
        messages.append({"role": "system", "content": str(hint)})

    messages.append({
        "role": "user",
        "content": f"Question: {question}\nReturn ONLY the JSON plan described above."
    })
    return messages


def _coerce_text(resp: Any) -> str:
    if resp is None:
        return ""
    if hasattr(resp, "content") and isinstance(getattr(resp, "content"), str):
        return resp.content
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            pass
        if isinstance(resp.get("content"), str):
            return resp["content"]
    if isinstance(resp, str):
        return resp
    return str(resp)


def _extract_json(s: str) -> str:
    if not s:
        return ""
    i = s.find("{")
    j = s.rfind("}")
    return s[i:j + 1] if i >= 0 and j > i else ""


def _normalize_calls(calls_in: Any, question: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in (calls_in or []):
        if not isinstance(c, dict):
            continue
        dom = _normalize_domain(c.get("domain_id", ""))
        cap = str(c.get("capability", "")).strip()
        args = c.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        strat = _choose_strategy(question, c.get("strategy"))

        # clamp capabilities set (executor supports these)
        allowed_caps = {
            "get_field", "find_latest", "sum_where",
            "topk_by_sum", "list_where", "semantic_search",
            "compare_periods"
        }
        if cap not in allowed_caps:
            # last resort: treat as list_where to avoid crash
            cap = "list_where"

        out.append({"domain_id": dom, "capability": cap, "args": args, "strategy": strat})
    return out


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#

def llm_plan(question: str) -> Dict[str, Any]:
    """
    Rule-first plan. If no rule matches, fall back to LLM (constrained).
    Always returns a normalized plan with allowed domains and an explicit strategy.
    """
    pack = _read_pack()
    rules = _planner_rules(pack)

    # 1) Try rules
    plan = _plan_from_rules(question, rules)
    if plan:
        return plan

    # 2) LLM fallback
    ctx = build_context(intent="planner", question=question, plan=None)
    hint = build_hint_for_question(question)
    messages = _build_llm_messages(ctx, pack, question, hint)

    llm = RUNTIME.chat()
    try:
        resp = llm.invoke(messages)  # LangChain ChatOpenAI path
    except AttributeError:
        resp = getattr(llm, "complete")(messages, temperature=0)  # custom client path

    text = _coerce_text(resp)
    js = _extract_json(text)

    try:
        obj = json.loads(js)
    except Exception:
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": [], "strategy": "deterministic"}

    if not isinstance(obj, dict):
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": [], "strategy": "deterministic"}

    obj.setdefault("intent", "unknown")
    obj.setdefault("must_produce", [])
    obj.setdefault("risk_if_missing", [])

    calls = _normalize_calls(obj.get("calls"), question)
    obj["calls"] = calls

    # plan-level strategy mirrors first call (or heuristic)
    plan_strategy = _choose_strategy(question, obj.get("strategy"))
    if calls:
        plan_strategy = calls[0].get("strategy", plan_strategy)
    obj["strategy"] = plan_strategy

    # Final domain clamp (paranoia)
    for c in obj["calls"]:
        c["domain_id"] = _normalize_domain(c.get("domain_id", ""))

    return obj