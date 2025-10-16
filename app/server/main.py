import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.orchestrator.planner import make_plan
from core.orchestrator.execute import execute_calls
from core.orchestrator.compose import compose_answer
from core.retrieval.rag_chain import unified_rag_answer_stream_events

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


def _read_cfg() -> dict:
    if not CONFIG_FILE.exists(): return {}
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}

def _sse(event: str, data: Any) -> bytes:
    """Pack an SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

def _ndj(obj: Dict[str, Any]) -> bytes:
    """Pack an NDJSON line."""
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


@APP.post("/api/ask/stream")
def ask_ndjson(payload: Dict[str, Any]):
    """
    Same as SSE but using NDJSON. Media type: application/x-ndjson
    """
    question   = payload.get("question") or ""
    account_id = payload.get("accountId")
    session_id = payload.get("sessionId") or str(uuid.uuid4())
    top_k      = int(payload.get("top_k") or 6)

    cfg = _read_cfg()

    def gen():
        plan = make_plan(question, str(CONFIG_FILE))
        yield _ndj({"type": "plan", "data": plan})

        calls = plan.get("calls", []) or []
        if len(calls) == 1 and (calls[0].get("domain_id") or "").lower() == "rag":
            for ev in unified_rag_answer_stream_events(
                question=question,
                session_id=session_id,
                account_id=account_id,
                cfg=cfg,
                k=top_k,
            ):
                yield _ndj(ev)
            yield _ndj({"type": "meta", "data": {"intent": plan.get("intent"), "sessionId": session_id}})
            return

        results: Dict[str, Any] = {}
        for call in calls:
            part = execute_calls([call], {
                "app_yaml": str(CONFIG_FILE),
                "account_id": account_id,
                "intent": plan.get("intent"),
                "question": question,
                "session_id": session_id,
                "top_k": top_k,
            })
            results.update(part)
            if part:
                k = next(iter(part.keys()))
                yield _ndj({"type": "result", "key": k, "data": part[k]})

        final = compose_answer(question, plan, results)
        yield _ndj({"type": "done", "data": {"answer": final, "sessionId": session_id}})

    return StreamingResponse(gen(), media_type="application/x-ndjson")

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