# core/semantics/enrich.py
from __future__ import annotations
from typing import Dict, Any
from datetime import datetime, timezone
from .temporal import resolve_period
from .synonyms import guess_category_from_text

def enrich_plan_with_semantics(plan: Dict[str, Any], question: str) -> Dict[str, Any]:
    q = (question or "").lower()
    time_hint = resolve_period(q, today=datetime.now(timezone.utc))
    cat_hint = guess_category_from_text(q)

    calls = plan.get("calls") or []
    for c in calls:
        args = c.setdefault("args", {})
        # fill period if missing
        if not any(k in args for k in ("period","period_start","period_end")):
            args.update(time_hint)
        # fill category if missing
        if cat_hint and "category" not in args:
            args["category"] = cat_hint

        # superlative hints
        if any(w in q for w in ["biggest","largest","highest","max"]):
            if c["domain_id"] == "transactions":
                c["capability"] = "max_amount"
                args.setdefault("top", 1)
        if any(w in q for w in ["compare","more than","less than","vs","versus"]):
            # let composer/executor interpret; leave as is unless you want to split periods here
            pass

    return plan