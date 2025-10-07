# core/semantics/synonyms.py
from __future__ import annotations

CATEGORY_SYNONYMS = {
    "travel": {"airbnb","uber","delta","united","marriott","airlines","hotel"},
    "groceries": {"whole foods","trader joe","kroger","safeway","aldi"},
    "electronics": {"apple","best buy","bose","amazon"},
    "dining": {"chipotle","mcdonald","starbucks","restaurant","dining"},
    "ecommerce": {"amazon","etsy","ebay"},
}

MERCHANT_CANON = {
    "apple store": "apple",
    "apple": "apple",
    "bestbuy": "best buy",
    "wholefoods": "whole foods",
}

def canonical_merchant(name: str | None) -> str:
    if not name: return "unknown"
    n = name.strip().lower()
    return MERCHANT_CANON.get(n, n)

def guess_category(merchant: str | None) -> str:
    m = canonical_merchant(merchant)
    for cat, vocab in CATEGORY_SYNONYMS.items():
        for v in vocab:
            if v in m:
                return cat
    # simple fallback
    if "amazon" in m: return "ecommerce"
    return "other"

def guess_category_from_text(text: str) -> str | None:
    t = (text or "").lower()
    for cat, vocab in CATEGORY_SYNONYMS.items():
        if any(v in t for v in vocab) or cat in t:
            return cat
    return None