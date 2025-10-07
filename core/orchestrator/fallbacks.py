# core/orchestrator/fallbacks.py
from __future__ import annotations
from typing import Dict, Any, Callable, List

# ---- How matching works ------------------------------------------------------
# Each fallback has:
#  - name: identifier
#  - match: function(question:str, plan:dict) -> bool   (True if this fallback should run)
#  - build: function(question:str) -> dict              (returns a valid Plan JSON)

def _match_interest(question: str, plan: Dict[str, Any]) -> bool:
    if plan.get("calls"):  # planner already produced something
        return False
    q = (question or "").lower()
    return "interest" in q  # broad match; refine if needed

def _build_interest_plan(question: str) -> Dict[str, Any]:
    # Non-zero latest interest fallback
    return {
        "intent": "last_interest",
        "calls": [
            {"domain_id":"statements","capability":"total_interest","args":{"period": None, "nonzero": True}},
            {"domain_id":"statements","capability":"interest_breakdown","args":{"period": None, "nonzero": True}},
        ],
        "must_produce": [],
        "risk_if_missing": ["may not explain trailing interest without breakdown"],
    }

def _match_balance(question: str, plan: Dict[str, Any]) -> bool:
    if plan.get("calls"):
        return False
    q = (question or "").lower()
    return any(w in q for w in ["balance", "current balance", "card balance"])

def _build_balance_plan(question: str) -> Dict[str, Any]:
    return {
        "intent": "get_current_balance",
        "calls": [
            {"domain_id":"account_summary","capability":"current_balance","args":{}}
        ],
        "must_produce": [],
        "risk_if_missing": [],
    }

def _match_available_credit(question: str, plan: Dict[str, Any]) -> bool:
    if plan.get("calls"):
        return False
    q = (question or "").lower()
    return any(w in q for w in ["available credit", "credit limit", "limit"])

def _build_available_credit_plan(question: str) -> Dict[str, Any]:
    return {
        "intent": "get_available_credit",
        "calls": [
            {"domain_id":"account_summary","capability":"available_credit","args":{}}
        ],
        "must_produce": [],
        "risk_if_missing": [],
    }

def _match_last_payment(question: str, plan: Dict[str, Any]) -> bool:
    if plan.get("calls"):
        return False
    q = (question or "").lower()
    return "last payment" in q or "most recent payment" in q or "when did i pay" in q

def _build_last_payment_plan(question: str) -> Dict[str, Any]:
    return {
        "intent": "last_payment",
        "calls": [
            {"domain_id":"payments","capability":"last_payment","args":{}}
        ],
        "must_produce": [],
        "risk_if_missing": [],
    }

def _match_biggest_purchase(question: str, plan: Dict[str, Any]) -> bool:
    return (not plan.get("calls")) and any(w in question.lower() for w in ["biggest", "largest", "highest"]) and "purchase" in question.lower()

def _build_biggest_purchase_plan(question: str) -> Dict[str, Any]:
    return {
        "intent": "max_purchase",
        "calls": [
            {"domain_id":"transactions","capability":"max_amount","args":{"period": None, "top": 1}}
        ],
        "must_produce": [],
        "risk_if_missing": [],
    }

# Ordered list = priority. First match wins.
FALLBACKS: List[Dict[str, Callable]] = [
    {"name": "interest_latest_nonzero", "match": _match_interest, "build": _build_interest_plan},
    {"name": "balance_simple",          "match": _match_balance,  "build": _build_balance_plan},
    {"name": "available_credit_simple", "match": _match_available_credit, "build": _build_available_credit_plan},
    {"name": "last_payment_simple",     "match": _match_last_payment, "build": _build_last_payment_plan},
]

def apply_fallbacks(question: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return the original plan if it's valid; otherwise apply first matching fallback."""
    if plan and plan.get("calls"):
        return plan
    for fb in FALLBACKS:
        try:
            if fb["match"](question, plan or {}):
                return fb["build"](question)
        except Exception:
            # Keep going; a single bad matcher shouldn't break the chain
            continue
    # No fallback matched; return the original plan (likely empty)
    return plan or {"intent":"unknown","calls":[], "must_produce":[], "risk_if_missing":[]}
