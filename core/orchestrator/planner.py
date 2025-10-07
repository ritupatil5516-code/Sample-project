import yaml
from .plan_llm import llm_plan
# optional: if you added the semantic enrich step earlier
try:
    from core.semantics.enrich import enrich_plan_with_semantics
except Exception:
    enrich_plan_with_semantics = lambda plan, q: plan

from .fallbacks import apply_fallbacks

def make_plan(question: str, app_yaml_path: str):
    cfg = yaml.safe_load(open(app_yaml_path).read())
    mode = cfg.get("planner_mode","llm")
    if mode == "llm":
        plan = llm_plan(question) or {"intent":"unknown","calls":[], "must_produce":[], "risk_if_missing":[]}
        plan = enrich_plan_with_semantics(plan, question)  # no-op if not installed
        # plan = apply_fallbacks(question, plan) ==> this is when LLM doesnt answer, and planner failure, so we fall back =>
        # Use Fallbacks: only when no plan exists → guarantee a safe, deterministic plan for common intents
        return plan
    return {"intent":"unknown","calls":[], "must_produce":[], "risk_if_missing":[]}