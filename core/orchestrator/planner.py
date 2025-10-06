import yaml
from .plan_llm import llm_plan

def make_plan(question: str, app_yaml_path: str):
    cfg = yaml.safe_load(open(app_yaml_path).read())
    mode = cfg.get("planner_mode","llm")
    if mode == "llm":
        plan = llm_plan(question)
        if (not plan.get("calls")) and ("interest" in question.lower()):
            return {"intent":"last_interest","calls":[
                {"domain_id":"statements","capability":"total_interest","args":{"period": None, "nonzero": True}},
                {"domain_id":"statements","capability":"interest_breakdown","args":{"period": None, "nonzero": True}}
            ],"must_produce":[],"risk_if_missing":[]}
        return plan
    return {"intent":"unknown","calls":[],"must_produce":[],"risk_if_missing":[]}
