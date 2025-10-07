# 💳 Credit Card Copilot — Full Working Repo

**Config**
- LLM (reasoning): OpenAI `gpt-4o-mini` (pluggable to Llama 70B)
- Embeddings: OpenAI `text-embedding-3-large` (pluggable to Qwen3)
- Run modes: Streamlit UI + FastAPI
- Policy PDF path: `data/agreement/Apple-Card-Customer-Agreement.pdf` (add yourself)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."

# Streamlit UI
streamlit run app/streamlit/app.py

# Optional FastAPI
# uvicorn app.server.main:app --reload --port 8001
```
