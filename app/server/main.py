import uvicorn
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
  results = execute_calls(
      plan.get("calls", []),
      {
          "app_yaml": "config/app.yaml",
          "account_id": req.accountId,
          "intent": plan.get("intent"),
          "question": req.question,  # << add
          "session_id": req.sessionId or "web"  # << add (any stable per-user/thread id)
      }
  )
  ans = compose_answer(req.question, plan, results)
  return {"plan": plan, "results": results, "answer": ans}

# server handler (pseudo)
from core.orchestrator.plan_llm import make_plan
from core.orchestrator.execute import execute_calls
from core.orchestrator.compose_answer import compose_answer

def answer(question: str, session_id: str, account_id: str | None = None):
    plan = make_plan(question, "config/app.yaml")
    results = execute_calls(
        plan.get("calls", []),
        {
            "app_yaml": "config/app.yaml",
            "session_id": session_id,
            "account_id": account_id,
            "question": question,  # so RAG has the original question
        },
    )
    ans = compose_answer(question, plan, results)
    return {"plan": plan, "results": results, "answer": ans}

if __name__ == "__main__":
    question = "when was i last charged interest ?"
    sessionId=""
    accountId="0269"
    plan = make_plan(question, "config/app.yaml")
    results = execute_calls(
        plan.get("calls", []),
        {
            "app_yaml": "config/app.yaml",
            "account_id": accountId,
            "intent": plan.get("intent"),
            "question": question,  # << add
            "session_id": sessionId or "web"  # << add (any stable per-user/thread id)
        }
    )
    ans = compose_answer(question, plan, results)
    print(ans)

from core.retrieval.index_builder import build_all_indexes


if __name__ == "__main__":
    print("[BOOT] Building all indexes…")
    build_all_indexes()
    print("[BOOT] Index build complete. Starting server…")
    # start your app (streamlit/fastapi/etc.)