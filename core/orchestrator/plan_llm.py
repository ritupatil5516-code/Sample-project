import json, yaml
from core.llm.providers import build_llm_from_config
from core.context.builder import build_context

SYSTEM = (
  "You are a planner for a credit-card copilot. "
  "Given a user question, output STRICT JSON: "
  "{intent: string, calls: [{domain_id, capability, args}], must_produce:[], risk_if_missing:[]}. "
  "Domains/capabilities: "
  "transactions:{spend_in_period, list_over_threshold, purchases_in_cycle}; "
  "payments:{last_payment, total_credited_year, payments_in_period}; "
  "statements:{total_interest, interest_breakdown, trailing_interest}; "
  "account_summary:{current_balance, available_credit}. "
  "If the user asks for latest/last/recent interest or omits period, set args.period to null. "
  "If the user says 'last interest' or 'most recent interest charged', also set args.nonzero=true "
  "to fetch the most recent period with non-zero interest. "
  "Return ONLY JSON."
)

EXAMPLE = {
  "intent": "last_interest",
  "calls": [
    {"domain_id":"statements","capability":"total_interest","args":{"period":None,"nonzero":True}},
    {"domain_id":"statements","capability":"interest_breakdown","args":{"period":None,"nonzero":True}}
  ],
  "must_produce": [],
  "risk_if_missing": []
}

def llm_plan(question: str):
    cfg = yaml.safe_load(open("config/app.yaml").read())
    llm = build_llm_from_config(cfg)

    ctx = build_context(intent="planner", question=question, plan=None)
    messages = [
        {"role":"system","content": ctx["system"]},
    ] + ctx["context_msgs"] + [
        {"role":"system","content": SYSTEM},
        {"role":"user","content": "Question: " + question + "\nReturn ONLY JSON. Example: " + json.dumps(EXAMPLE)}
    ]
    txt = llm.complete(messages, model=cfg["llm"]["model"], temperature=0)
    start = txt.find("{"); end = txt.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        txt = txt[start:end+1]
    try:
        obj = json.loads(txt)
        obj.setdefault("calls", []); obj.setdefault("must_produce", []); obj.setdefault("risk_if_missing", [])
        for c in obj["calls"]: c.setdefault("args", {})
        return obj
    except Exception:
        return {"intent":"unknown","calls":[], "must_produce":[], "risk_if_missing":[]}
