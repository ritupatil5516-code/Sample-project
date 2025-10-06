from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import yaml
from core.retrieval.policy_index import get_policy_snippet

PKG = Path("core/context/packs")
CFG = Path("config/context.yaml")

def _load_cfg() -> dict:
    if CFG.exists():
        return yaml.safe_load(CFG.read_text()) or {}
    return {"enabled": True, "packs":{}, "budget":{}, "retrieval":{}}

def _read_pack(name: str) -> str:
    p = PKG / f"{name}.md"
    return p.read_text() if p.exists() else ""

def _cap_budget(texts: List[str], soft_cap: int) -> List[str]:
    out, total = [], 0
    for t in texts:
        if total + len(t) > soft_cap: break
        out.append(t); total += len(t)
    return out

def DEFAULT_SYSTEM() -> str:
    return ("You are a credit-card copilot planner/composer. "
            "Use ONLY provided data/calculator outputs for numbers. "
            "Do not invent amounts, dates, or periods. "
            "Prefer domain field names exactly as shown in schemas. "
            "For policy explanations, use retrieved policy snippets; do not fabricate rules.")

def build_context(intent: str, question: str, plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = _load_cfg()
    if not cfg.get("enabled", True):
        return {"system": DEFAULT_SYSTEM(), "context_msgs": []}

    system = DEFAULT_SYSTEM()
    packs_cfg = cfg.get("packs", {})
    selected = list(packs_cfg.get("common", []))

    low = ((intent or "") + " " + question).lower()
    if any(w in low for w in ["interest", "apr", "grace", "trailing"]):
        selected += packs_cfg.get("interest", [])
    if any(w in low for w in ["balance", "available credit", "limit"]):
        selected += packs_cfg.get("balance", [])
    if any(w in low for w in ["transaction", "spend", "over $", "over$", "merchant"]):
        selected += packs_cfg.get("transactions", [])

    sections = []
    for name in selected:
        txt = _read_pack(name)
        if txt.strip():
            sections.append(f"## {name.replace('_',' ').title()}\n{txt.strip()}")

    retrieval_cfg = cfg.get("retrieval", {})
    need_policy = any(w in low for w in ["why", "interest", "grace", "trailing"])
    if need_policy:
        q = question + (" grace period daily periodic rate trailing interest billing cycle" if retrieval_cfg.get("policy_query_expansion", True) else "")
        res = get_policy_snippet(q) or {}
        snippet = res.get("snippet") or ""
        if len(snippet) >= retrieval_cfg.get("policy_min_chars", 400):
            sections.append("## Policy Snippets\n" + snippet)

    per_section_soft = int(cfg.get("budget", {}).get("per_section_soft", 800))
    sections = _cap_budget(sections, per_section_soft * len(sections))
    context_msgs = [{"role":"assistant","content": s} for s in sections]

    return {"system": system, "context_msgs": context_msgs}
