# core/context/hints.py
from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path
import yaml

PACK_PATH = Path("core/context/packs/core.yaml")

def _load_synonyms() -> Dict[str, list]:
    try:
        pack = yaml.safe_load(PACK_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pack = {}
    pr = (pack.get("planner_rules") or {}) if isinstance(pack, dict) else {}
    syn = pr.get("synonyms") or {}
    # normalize to sets of lower strings
    return {k: [str(x).lower() for x in (v or [])] for k, v in syn.items()}

def _has_any(text: str, keys: list) -> bool:
    return any(k in text for k in (keys or []))

def build_hint_for_question(q: str) -> Optional[Dict[str, str]]:
    """
    Hints derived from planner_rules.synonyms (if present) so they stay in sync with core.yaml.
    Falls back to light heuristics if synonyms are missing.
    """
    ql = (q or "").lower()
    syn = _load_synonyms()

    hints = []

    # Recency routing (map to correct timestamp by domain)
    recency_words = syn.get("recency", ["last", "latest", "recent", "most recent", "recently"])
    tx_words      = syn.get("tx_words", ["transaction", "transactions", "purchase", "purchases", "charge", "charges"])
    pay_words     = syn.get("payment_words", ["payment", "paid", "pay", "remit"])
    stmt_words    = syn.get("stmt_words", ["statement", "cycle", "period", "closing"])
    interest_words= syn.get("interest", ["interest", "finance charge", "trailing interest"])

    if _has_any(ql, recency_words) and _has_any(ql, tx_words):
        hints.append("For recency over transactions, use the maximum postedDateTime.")
    if _has_any(ql, recency_words) and _has_any(ql, pay_words):
        hints.append("For recency over payments, use the maximum paymentPostedDateTime.")
    if _has_any(ql, recency_words) and _has_any(ql, stmt_words):
        hints.append("For recency over statements, use the maximum closingDateTime.")

    # Interest guidance
    if _has_any(ql, interest_words):
        hints.append("Interest amounts are in statements.interestCharged; prefer the latest statement with interestCharged > 0, otherwise the latest period.")

    # Balance/credit nudges
    if "available credit" in ql or "utilization" in ql or "current balance" in ql:
        hints.append("Use account summary for currentBalance, availableCredit, and derived utilization.")

    if not hints:
        return None
    return {"role": "system", "content": "Hints:\n- " + "\n- ".join(hints)}