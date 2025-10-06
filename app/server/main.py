from fastapi import FastAPI
from pydantic import BaseModel
from core.orchestrator.planner import make_plan
from core.orchestrator.execute import execute_calls
from core.orchestrator.compose import compose_answer

app = FastAPI()

class AskReq(BaseModel):
    session_id: str
    question: str

@app.post("/ask")
def ask(req: AskReq):
    plan = make_plan(req.question, "config/app.yaml")
    results = execute_calls(plan.get("calls", []), {"app_yaml":"config/app.yaml","policy_store_dir":"var/policies","intent":plan.get("intent")})
    ans = compose_answer(req.question, plan, results)
    return {"plan": plan, "results": results, "answer": ans}
