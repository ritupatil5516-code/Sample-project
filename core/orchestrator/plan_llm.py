# core/orchestrator/plan_llm.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Use the one shared LLM configured at startup
from src.core.runtime import RUNTIME

# Optional (nice-to-have) context packs; safely ignored if not present
try:
    from core.context.builder import build_context  # type: ignore
except Exception:  # pragma: no cover
    def build_context(intent: str, question: str, plan: Optional[dict]) -> Dict[str, Any]:
        return {"system": "You are a planner for a credit-card copilot.", "context_msgs": []}

try:
    from core.context.hints import build_hint_for_question  # type: ignore
except Exception:  # pragma: no cover
    def build_hint_for_question(q: str) -> str:
        return ""


# ------------------------------ Planning contract ------------------------------

SYSTEM_CONTRACT = (
    "You are a planner for a credit-card copilot.\n"
    "Given a user question, return STRICT JSON with this shape:\n"
    "{\n"
    '  "intent": string,\n'
    '  "calls": [\n'
    "    {\n"
    '      "domain_id": string,             // one of: "transactions", "payments", "statements", "accounts", or "rag"\n'
    '      "capability": string,            // deterministic DSL ops: "get_field", "find_latest", "sum_where", "topk_by_sum", "list_where";\n'
    '                                       // if domain_id == "rag", capability can be "unified_rag" and args can include {"account_id": "..."}\n'
    '      "args": object,                  // op-specific args (see below)\n'
    '      "strategy": "deterministic" | "rag" | "auto"\n'
    "    }\n"
    "  ],\n"
    '  "must_produce": [],\n'
    '  "risk_if_missing": []\n'
    "}\n\n"
    "Rules:\n"
    "- Prefer strategy = \"deterministic\" when the question maps to exact fields or precise tabular logic.\n"
    "- Use strategy = \"rag\" for policy/handbook questions, fuzzy reasoning, or multi-document explanations.\n"
    "- Use strategy = \"auto\" if you are unsure; executor will try deterministic first then RAG fallback.\n"
    "- domains:\n"
    "  * accounts (dict-shaped): fields like accountStatus, currentBalance, availableCredit, creditLimit\n"
    "  * transactions/payments/statements (list-shaped)\n"
    "- deterministic DSL ops & typical args:\n"
    "  * get_field     {\"field\": \"accountStatus\"}\n"
    "  * find_latest   {\"field\": \"amount\", \"where\": {\"merchant\": \"apple*\"}}\n"
    "  * sum_where     {\"sum_field\": \"amount\", \"where\": {\"category\": \"GROCERIES\"}}\n"
    "  * topk_by_sum   {\"key_field\": \"merchantName\", \"sum_field\": \"amount\", \"where\": {...}, \"k\": 5}\n"
    "  * list_where    {\"where\": {...}, \"per_key\": \"merchantName\", \"limit_per_key\": 2}\n"
    "- Aliases you may reference in args.where or field names: status→accountStatus, balance→currentBalance, available→availableCredit, merchant→merchantName, date→postedDateTime.\n"
    "- For policy/handbook/fees/eligibility questions, either set domain_id=\"rag\" with capability=\"unified_rag\" OR set a normal domain with strategy=\"rag\".\n"
    "- Return ONLY JSON (no markdown, no prose, no code fences)."
)

EXAMPLES: List[Dict[str, Any]] = [
    # 1) Direct account field → deterministic
    {
        "intent": "account_status",
        "calls": [
            {"domain_id": "accounts", "capability": "get_field",
             "args": {"field": "accountStatus"}, "strategy": "deterministic"}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    # 2) Top merchants (12 months) → deterministic aggregation
    {
        "intent": "top_merchants_12m",
        "calls": [
            {"domain_id": "transactions", "capability": "topk_by_sum",
             "args": {"key_field": "merchantName", "sum_field": "amount",
                      "where": {"status": "POSTED"}, "k": 5},
             "strategy": "deterministic"}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    # 3) Policy/handbook question → RAG
    {
        "intent": "late_fee_policy",
        "calls": [
            {"domain_id": "rag", "capability": "unified_rag",
             "args": {"account_id": null, "k": 6}, "strategy": "rag"}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
    # 4) Ambiguous → auto (executor may fallback to RAG)
    {
        "intent": "spend_pattern",
        "calls": [
            {"domain_id": "transactions", "capability": "list_where",
             "args": {"where": {"amount": {"$gte": 100}}}, "strategy": "auto"}
        ],
        "must_produce": [],
        "risk_if_missing": []
    },
]


def _examples_block() -> str:
    return "\n".join(json.dumps(e, separators=(",", ":")) for e in EXAMPLES)


# ------------------------------ Utilities ------------------------------

def _outer_json(text: str) -> str:
    """Extract outermost {...} to survive accidental extra tokens or code fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        return text[start : end + 1]
    return text.strip()

def _shape_plan(obj: Any) -> Dict[str, Any]:
    """Ensure the planner output has the expected shape & defaults."""
    if not isinstance(obj, dict):
        return {"intent": "unknown", "calls": [], "must_produce": [], "risk_if_missing": []}

    intent = obj.get("intent") or "unknown"
    must = obj.get("must_produce") or []
    risk = obj.get("risk_if_missing") or []

    shaped_calls: List[Dict[str, Any]] = []
    for c in obj.get("calls") or []:
        if not isinstance(c, dict):
            continue
        dom = str(c.get("domain_id", "")).strip().lower().replace("-", "_")
        if dom in ("account_summary", "account"):  # normalize
            dom = "accounts"
        cap = str(c.get("capability", "")).strip().lower().replace("-", "_")
        args = c.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        strategy = str(c.get("strategy") or "auto").strip().lower()
        if strategy not in ("deterministic", "rag", "auto"):
            strategy = "auto"
        shaped_calls.append({"domain_id": dom, "capability": cap, "args": args, "strategy": strategy})

    return {
        "intent": intent,
        "calls": shaped_calls,
        "must_produce": must if isinstance(must, list) else [],
        "risk_if_missing": risk if isinstance(risk, list) else [],
    }


# ------------------------------ Heuristic fallback (no-LLM) ------------------------------

_KEY_PATTS = [
    (re.compile(r"\b(account\s*status)\b", re.I), lambda: {
        "intent": "account_status",
        "calls": [{"domain_id": "accounts", "capability": "get_field",
                   "args": {"field": "accountStatus"}, "strategy": "deterministic"}],
        "must_produce": [], "risk_if_missing": []
    }),
    (re.compile(r"\b(current\s*balance|available\s*credit)\b", re.I), lambda: {
        "intent": "account_balance_or_available",
        "calls": [{"domain_id": "accounts", "capability": "get_field",
                   "args": {"field": "currentBalance"}, "strategy": "deterministic"}],
        "must_produce": [], "risk_if_missing": []
    }),
    (re.compile(r"\b(where|did)\s+.*\b(spend|most)\b", re.I), lambda: {
        "intent": "top_merchants",
        "calls": [{"domain_id": "transactions", "capability": "topk_by_sum",
                   "args": {"key_field": "merchantName", "sum_field": "amount",
                            "where": {"status": "POSTED"}, "k": 5},
                   "strategy": "deterministic"}],
        "must_produce": [], "risk_if_missing": []
    }),
    (re.compile(r"\b(late fee|policy|agreement|handbook)\b", re.I), lambda: {
        "intent": "policy_lookup",
        "calls": [{"domain_id": "rag", "capability": "unified_rag",
                   "args": {"account_id": None, "k": 6}, "strategy": "rag"}],
        "must_produce": [], "risk_if_missing": []
    }),
]


def _fallback_plan(question: str) -> Dict[str, Any]:
    for patt, make in _KEY_PATTS:
        if patt.search(question or ""):
            return make()
    # last resort
    return {
        "intent": "unknown",
        "calls": [{"domain_id": "rag", "capability": "unified_rag",
                   "args": {"account_id": None, "k": 6}, "strategy": "auto"}],
        "must_produce": [], "risk_if_missing": []
    }


# ------------------------------ Public API ------------------------------

def llm_plan(question: str) -> Dict[str, Any]:
    """
    Use the shared Chat LLM (RUNTIME.chat()) to produce a plan.
    Falls back to simple heuristics if JSON parsing fails.
    """
    chat = RUNTIME.chat()  # langchain_openai.ChatOpenAI (already configured at startup)

    # Optional contextual packs
    ctx = build_context(intent="planner", question=question, plan=None)
    hint = build_hint_for_question(question)
    ctx_msgs = ctx.get("context_msgs") or []

    # Build messages (LangChain Message objects are fine, but ChatOpenAI also accepts dicts)
    messages: List[Dict[str, str]] = []
    if ctx.get("system"):
        messages.append({"role": "system", "content": str(ctx["system"])})
    for m in ctx_msgs:
        if isinstance(m, dict) and m.get("role") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    if hint:
        messages.append({"role": "system", "content": hint})
    messages.append({"role": "system", "content": SYSTEM_CONTRACT})
    messages.append({
        "role": "user",
        "content": (
            "Return ONLY JSON. Do not include explanations or markdown code fences.\n\n"
            f"Examples:\n{_examples_block()}\n\n"
            f"Question: {question}"
        ),
    })

    # Call the shared LLM
    resp = chat.invoke(messages)
    raw = getattr(resp, "content", "") or str(resp)
    text = _outer_json(raw)

    try:
        obj = json.loads(text)
        return _shape_plan(obj)
    except Exception:
        # Heuristic fallback to keep UX unblocked
        return _fallback_plan(question)


# Backwards-compatible entry point (your server calls make_plan(...))
def make_plan(question: str, app_yaml_path: str = "config/app.yaml") -> Dict[str, Any]:
    # We no longer need app_yaml_path here because RUNTIME holds config + clients.
    # Kept in signature for compatibility with existing server code.
    return llm_plan(question)