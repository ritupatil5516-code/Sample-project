from typing import Optional, Dict, Any
import yaml
from .policy_faiss import ensure_policy_index as _ensure_policy_index, query_policy as _query_policy

def _policy_enabled() -> bool:
    cfg = yaml.safe_load(open("config/app.yaml").read())
    return (cfg.get("policy") or {}).get("enabled", False)

def ensure_policy_index() -> None:
    if _policy_enabled():
        _ensure_policy_index()

def get_policy_snippet(capability: str, store_dir: Optional[str] = None) -> Dict[str, Any]:
    if not _policy_enabled():
        return {"snippet":"", "citation": None, "error": "policy_disabled"}
    try:
        _ensure_policy_index()
        res = _query_policy(capability, store_dir=store_dir, top_k=3) or {}
        return {"snippet": res.get("snippet", ""), "citation": res.get("citation"),
                "snippets": res.get("snippets"), "ids": res.get("ids"), "scores": res.get("scores")}
    except Exception as e:
        return {"snippet":"", "citation": None, "error": f"policy_lookup_failed: {e}"}
