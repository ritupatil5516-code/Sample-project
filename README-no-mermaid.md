
# Agent Desktop Assist — Architecture (Mermaid‑free)

This version of the README removes Mermaid diagrams and uses ASCII diagrams so it renders everywhere (JetBrains, GitHub raw, terminals). If you want Mermaid back, see **“How to view Mermaid diagrams”** below.

---

## Architecture (ASCII)

```
+--------------------------+
| Ingestion & Indexing     |
+--------------------------+
| JSON sources per account |
|  - transactions.json     |
|  - payments.json         |
|  - statements.json       |
|  - account_summary.json  |
| Knowledge sources        |
|  - data/knowledge/*.md   |
|  - data/agreement/*.pdf  |
+--------------------------+
            |
            v
+----------------------------+
| index_builder.py           |
|  - renders rich text rows  |
|  - adds helpful metadata   |
|  - appends RAW_JSON tail   |
|  - embeds via HF model     |
|  - writes FAISS stores     |
+----------------------------+
      |                         \
      |                          \
      v                           v
+----------------------------+   +--------------------------+
| indexesstore/accounts/<id> |   | indexesstore/knowledge  |
|    (faiss + *.json)        |   |    (faiss + *.json)     |
+----------------------------+   +--------------------------+
```

```
+---------------------------------------------------------+
| Runtime                                                 |
+---------------------------------------------------------+
| 1) User asks a question                                 |
| 2) Planner (core.yaml) picks a path:                    |
|    a) Deterministic data call(s)                        |
|       - dsl_ops.py: get_field, find_latest, list_where  |
|       - Domains load JSON rows                          |
|       - Orchestrator executes + compose answer          |
|                                                         |
|    b) RAG explanation                                   |
|       - rag_chain.py retrieves from FAISS indexes       |
|       - RUNTIME.llm synthesizes ONLY from snippets      |
|       - Answer + sources returned                       |
+---------------------------------------------------------+
```

---

## Major Components

### 1) `index_builder.py` — ingestion & vector indexes
- Reads account JSONs and knowledge docs
- Renders compact, semantically rich text per row (plus `RAW_JSON`)
- Adds metadata (domain, ts, ym, amounts, interest flags, etc.)
- Embeds with `sentence-transformers/all-MiniLM-L6-v2` (default)
- Persists FAISS stores under:
  - `src/api/contextApp/indexesstore/accounts/<accountId>/llama`
  - `src/api/contextApp/indexesstore/knowledge/llama`

**Why FAISS + local embeddings?**  
Fast, offline-friendly, reproducible. You can swap the embedder by setting `Settings.embed_model` at runtime without changing the indexing pipeline.

---

### 2) `core.yaml` — planner & routes
- **System + Reasoning**: guardrails (“sort timestamps for *latest*”, etc.)
- **Synonyms**: robust lexical matching (e.g., “recent”, “latest”, “last”)
- **Routes**: declarative mapping from user intent → deterministic ops or RAG
- **Deterministic first** for factual questions; **RAG** for “why/explain/policy”.

**Why a planner?**  
Repeatable, testable paths. Keeps “what/when/how much” answers consistent and auditable, while letting RAG handle narrative/explanations.

---

### 3) `dsl_ops.py` — deterministic execution
- Uniform set of functions operating on raw JSON:
  - `get_field`, `find_latest`, `list_where`, `sum_where`, `topk_by_sum`, `compare_periods`
  - `explain_interest` (combined statement + in-period txns/payments)
- Handles date parsing, period symbols (e.g., `LAST_MONTH`), filters, sorting.
- Returns structured payloads; **no LLM guessing** here.

**Why code, not prompts?**  
Money math and “latest” logic must be precise. This avoids model drift and makes unit tests easy.

---

### 4) `rag_chain.py` — retrieval + synthesis
- Loads LlamaIndex stores from `indexesstore`
- Retrieves top‑k nodes from:
  - Account index (if provided)
  - Knowledge index
- Builds a single context block
- Calls the LLM with “**answer ONLY from context**”
- Returns **answer + sources**

**Why a manual chain (vs heavy framework)?**  
A few clear steps (retrieve → pack context → synthesize) are easy to reason about and debug. You can later swap the LLM or rerankers without touching the planner or deterministic ops.

---

### 5) `RUNTIME` — singletons & wiring
- Provides: `cfg` (paths), `embedding()` (process‑wide embedder), `chat()` (LLM), `memory(session_id)`
- Centralizes model settings so both indexer & RAG use the same embeddings.

---

## How to view Mermaid diagrams (optional)

If you’d prefer the pretty Mermaid version of these diagrams:
1. **GitHub** renders Mermaid in README by default. Commit & view in the repo.
2. **VS Code**: install *“Markdown Preview Mermaid Support”* and open the Markdown preview (`Ctrl/Cmd+Shift+V`).
3. **JetBrains (PyCharm/IntelliJ)**: install a Mermaid renderer plugin (e.g., *“Mermaid”* or *“Kroki”* from the Marketplace), then open the Markdown preview.
4. **CLI** (to export .svg/.png):  
   ```bash
   npm i -g @mermaid-js/mermaid-cli
   mmdc -i diagrams/architecture.mmd -o diagrams/architecture.svg
   ```

> Tip: I can generate a `README-mermaid.md` that includes the original Mermaid blocks next to these ASCII fallbacks so it looks good on GitHub *and* in IDEs without Mermaid.

---

## FAQ

**Why do some RAG answers cite multiple sources?**  
We interleave account and knowledge hits. The LLM is instructed to answer *only* from the provided snippets; the source list is shown for auditability.

**Latest/Last sometimes picked the wrong row—what fixed it?**  
The planner sends an explicit `find_latest` with a date key and the executor sorts on that key; if it’s missing, it falls back to the first available timestamp among `postedDateTime`, `transactionDateTime`, `paymentPostedDateTime`, `paymentDateTime`, `closingDateTime`, `openingDateTime`, `date`.

**Can we extend this to new domains?**  
Yes. Add a loader + renderer in `index_builder.py` (to get good embeddings), declare routes in `core.yaml`, and implement any deterministic ops in `dsl_ops.py`.
