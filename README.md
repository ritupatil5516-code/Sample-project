# Credit Card Copilot (with Context Engineering)

LLM-assisted, deterministic credit-card Q&A across transactions, payments, statements, account summary.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
streamlit run app/streamlit/app.py
```
Policy PDF: put real Apple Card agreement at `data/agreement/Agreement.pdf` and set `policy.enabled: true` in `config/app.yaml`, then `rm -rf var/policies` and rerun.
