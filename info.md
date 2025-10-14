# Credit-Card Copilot

Production-ready overview of the **plan → execute → compose** architecture with **RAG** (retrieval-augmented generation) and **conversational memory**.

---

## 0) TL;DR

- **Planner LLM** turns a user question into a **strict JSON plan** (which tools to call).
- **Executor** runs either the **deterministic DSL** (direct JSON field/aggregate queries) or the **RAG lane** (semantic retrieval over account JSON + handbook/policy) with short-term memory.
- **Composer** renders heterogeneous results into a clean answer (tables, money/date formats, optional RAG sources).

---

## 1) High-Level Flow

```mermaid
flowchart LR
  A[User Question] --> B[plan_llm: Planner LLM]
  B -->|STRICT JSON plan| C[execute_calls: Executor]
  C -->|results dict| D[compose_answer: Presenter]
  D --> E[Final Answer]

  subgraph Structured Lane (DSL)
    C1[get_field / find_latest / sum_where / topk_by_sum / list_where / semantic_search]
  end

  subgraph RAG Lane (Conversational)
    R1[Account FAISS Index]
    R2[Knowledge FAISS Index]
    RC[ConversationalRetrievalChain + Memory]
  end

  C -->|domain_id != rag| C1
  C -->|domain_id == rag| RC
  RC --> R1
  RC --> R2

```
## 2) What the System Uses (and Why)
Component
Library/Tech
Why we use it
Planner LLM
OpenAI-compatible /chat/completions (e.g., gpt-4o-mini)
Deterministic routing: emits strict JSON plans (no hallucinated calculations).
DSL (structured lane)
Pure Python
Ultra-fast, safe answers for direct data questions (no embeddings).
RAG (semantic lane)
LlamaIndex + FAISS (indexing) + LangChain (memory & conversational chain)
Easy ingestion/persistence to FAISS + robust conversational retrieval with short-term memory.
Vector store
FAISS
Local, fast, portable ANN retrieval; persists on disk (indexesstore).
Memory
LangChain ConversationBufferWindowMemory(k=10)
Keeps last 10 turns per session for follow-ups.
Embeddings
OpenAI text-embedding-3-large (default)
High-quality semantic search; dim=3072 (important for FAISS).


