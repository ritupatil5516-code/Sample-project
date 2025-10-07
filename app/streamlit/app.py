import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

import json, requests, yaml, streamlit as st
from core.index.startup import build_on_startup
from core.orchestrator.planner import make_plan
from core.orchestrator.execute import execute_calls
from core.orchestrator.compose import compose_answer
from core.orchestrator.verify import verify
load_dotenv()
st.set_page_config(page_title="Credit Card Co-Pilot", page_icon="💳", layout="centered")
st.sidebar.caption(f"OPENAI key detected: {'yes' if (os.getenv('OPENAI_API_KEY') or '').strip() else 'no'}")
APP = yaml.safe_load(Path("config/app.yaml").read_text())
if "indexes_built" not in st.session_state:
    build_on_startup()
    st.session_state.indexes_built = True

st.sidebar.header("Settings")
planner_mode = st.sidebar.selectbox("Planner mode", ["llm","rule"], index=0 if APP.get("planner_mode")=="llm" else 1)
local_mode = st.sidebar.toggle("Local mode (bypass API)", value=APP.get("local_mode", True))
base_url = st.sidebar.text_input("API Base URL", value="http://localhost:8001")
session_id = st.sidebar.text_input("Session ID", value="demo")

if st.sidebar.button("Save settings"):
    cfg = APP.copy()
    cfg["planner_mode"] = planner_mode
    cfg["local_mode"] = bool(local_mode)
    Path("config/app.yaml").write_text(yaml.safe_dump(cfg))
    st.sidebar.success("Saved.")

st.markdown("### 💳 Credit Card Co-Pilot")

if "history" not in st.session_state: st.session_state.history = []

for msg in st.session_state.history[-10:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if meta.get("show_trace"):
            with st.expander("Trace & plan"):
                st.json({"plan": meta.get("plan"), "results": meta.get("results")})
        if meta.get("citations"):
            with st.expander("Citations"):
                st.write("; ".join(meta["citations"]))

def run_local(question: str):
    plan = make_plan(question, "config/app.yaml")
    results = execute_calls(plan.get("calls", []), {"app_yaml":"config/app.yaml","policy_store_dir":"var/policies","intent":plan.get("intent")})
    ok, missing = verify(plan, results)
    composed = compose_answer(question, plan, results)
    return {"plan": plan, "results": results, "ok": ok, "missing": missing, "answer": composed}

prompt = st.chat_input("Ask a question… e.g., When was I last charged non-zero interest?")
if prompt:
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        ph = st.empty(); ph.markdown("…thinking")
        try:
            out = run_local(prompt) if local_mode else requests.post(base_url.rstrip('/')+'/ask', json={"session_id":session_id,"question":prompt}, timeout=60).json()
            ans = out.get("answer", {}) or {}; text = ans.get("answer") or "No answer."
            ph.markdown(text)
            plan = out.get("plan"); results = out.get("results")
            with st.expander("Trace & plan"): st.json({"plan": plan, "results": results})
            st.session_state.history.append({"role":"user","content":prompt})
            st.session_state.history.append({"role":"assistant","content":text, "meta":{"show_trace": True, "plan": plan, "results": results}})
        except Exception as e:
            ph.markdown(f"⚠️ Error: {e}")
            st.session_state.history.append({"role":"user","content":prompt})
            st.session_state.history.append({"role":"assistant","content":f"⚠️ Error: {e}", "meta":{}})
