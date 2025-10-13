import os
import sys
import json
import requests
import yaml
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Core imports ---
from core.index.startup import build_on_startup
from core.orchestrator.planner import make_plan
from core.orchestrator.execute import execute_calls
from core.orchestrator.compose import compose_answer
from core.orchestrator.verify import verify

# --- Load environment ---
load_dotenv()

# -------------------- UI CONFIG --------------------
st.set_page_config(page_title="💳 Credit Card Co-Pilot", page_icon="💳", layout="centered")
st.sidebar.caption(f"🔑 OPENAI key detected: {'✅ yes' if (os.getenv('OPENAI_API_KEY') or '').strip() else '❌ no'}")

# --- Load config ---
APP = yaml.safe_load(Path("config/app.yaml").read_text())

# --- Build indexes once on startup ---
if "indexes_built" not in st.session_state:
    build_on_startup()
    st.session_state.indexes_built = True

# -------------------- SIDEBAR SETTINGS --------------------
st.sidebar.header("⚙️ Settings")
planner_mode = st.sidebar.selectbox("Planner mode", ["llm", "rule"], index=0 if APP.get("planner_mode") == "llm" else 1)
local_mode = st.sidebar.toggle("Local mode (bypass backend API)", value=APP.get("local_mode", True))
base_url = st.sidebar.text_input("API Base URL", value="http://localhost:8001")
session_id = st.sidebar.text_input("Session ID", value="demo")

if st.sidebar.button("💾 Save settings"):
    cfg = APP.copy()
    cfg["planner_mode"] = planner_mode
    cfg["local_mode"] = bool(local_mode)
    Path("config/app.yaml").write_text(yaml.safe_dump(cfg))
    st.sidebar.success("✅ Settings saved.")

# -------------------- MAIN HEADER --------------------
st.markdown("## 💳 Credit Card Co-Pilot")
st.caption("Ask anything about your credit card activity, interest, payments, or policy handbook.")

# -------------------- Chat History --------------------
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history[-10:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if meta.get("show_trace"):
            with st.expander("🔍 Trace & Plan"):
                st.json({"plan": meta.get("plan"), "results": meta.get("results")})
        if meta.get("citations"):
            with st.expander("📚 Citations"):
                st.write("; ".join(meta["citations"]))

# -------------------- Local execution helper --------------------
def run_local(question: str):
    plan = make_plan(question, "config/app.yaml")
    results = execute_calls(
        plan.get("calls", []),
        {"app_yaml": "config/app.yaml", "policy_store_dir": "var/policies", "intent": plan.get("intent")}
    )
    ok, missing = verify(plan, results)

    composed = compose_answer(question, plan, results)
    if isinstance(composed, dict):
        final_text = composed.get("answer") or composed.get("content") or str(composed)
    else:
        final_text = str(composed or "No information found.").strip()

    # Handle empty / unknown responses
    if not final_text or final_text.lower() in ["", "none", "null"]:
        final_text = "No information found."

    return {
        "plan": plan,
        "results": results,
        "ok": ok,
        "missing": missing,
        "answer": final_text
    }

# -------------------- MAIN INPUT --------------------
prompt = st.chat_input("💬 Ask a question… e.g., 'When was I last charged non-zero interest?'")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ph = st.empty()
        ph.markdown("🤔 Thinking...")
        try:
            if local_mode:
                out = run_local(prompt)
            else:
                response = requests.post(
                    base_url.rstrip('/') + '/ask',
                    json={"session_id": session_id, "question": prompt},
                    timeout=60
                )
                out = response.json()

            # Extract clean final answer (plain text only)
            # --- Force plain-text rendering ---
            raw_ans = out.get("answer")

            # If it's a dict: try to extract .answer or .content
            if isinstance(raw_ans, dict):
                raw_ans = raw_ans.get("answer") or raw_ans.get("content")

            # If it's still not a string: fallback
            if not isinstance(raw_ans, str) or raw_ans.strip() == "":
                raw_ans = "No information found."

            # Final safety: remove JSON-like wrappers if LLM returns a JSON string
            if raw_ans.strip().startswith("{") and '"answer":' in raw_ans:
                try:
                    parsed = json.loads(raw_ans)
                    raw_ans = parsed.get("answer", raw_ans)
                except Exception:
                    pass

            ph.markdown(raw_ans)

            # Trace & plan
            plan = out.get("plan")
            results = out.get("results")
            with st.expander("🔍 Trace & Plan"):
                st.json({"plan": plan, "results": results})

            # Save to chat history
            st.session_state.history.append({"role": "user", "content": prompt})
            st.session_state.history.append({
                "role": "assistant",
                "content": raw_ans,
                "meta": {"show_trace": True, "plan": plan, "results": results}
            })

        except Exception as e:
            error_msg = f"⚠️ Error: {e}"
            ph.markdown(error_msg)
            st.session_state.history.append({"role": "user", "content": prompt})
            st.session_state.history.append({"role": "assistant", "content": error_msg, "meta": {}})