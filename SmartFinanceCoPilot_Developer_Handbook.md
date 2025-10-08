# SmartFinanceCoPilot --- Developer & Architecture Handbook

## Table of Contents

1.  Introduction
2.  System Overview
3.  Architecture Diagram
4.  Core Components
5.  Data Flow
6.  Planner
7.  Executor
8.  FAISS Index Engine
9.  Temporal Registry
10. Policy Index
11. Context & Hint Packs
12. Domain Calculators
13. Semantic Search
14. Query Execution Flow
15. Error Handling
16. Deployment
17. Extending the System
18. Troubleshooting
19. Example Queries
20. Appendix: Mermaid Diagrams

------------------------------------------------------------------------

## 1. Introduction

SmartFinanceCoPilot is an AI-driven assistant that blends deterministic
data computation with large language model (LLM) reasoning. It answers
financial questions such as *"When was I last charged interest?"* or
*"Did I buy anything related to travel?"* using a combination of
rule-based and semantic retrieval.

------------------------------------------------------------------------

## 2. System Overview

SmartFinanceCoPilot consists of the following main layers:

-   **Streamlit UI:** Interactive frontend for user queries.
-   **Planner:** Uses an LLM to parse natural language into structured
    plans.
-   **Executor:** Executes those plans deterministically or via semantic
    similarity.
-   **Indexing Subsystem:** Prepares FAISS, temporal, and policy
    indexes.
-   **Domains:** Handle structured data like transactions, payments, and
    statements.

------------------------------------------------------------------------

## 3. Architecture Diagram

``` mermaid
flowchart LR
    User(UI) --> Planner(LLM Planner)
    Planner --> Executor
    Executor --> FAISS[FAISS Vector Store]
    Executor --> Temporal[Temporal Index]
    Executor --> Policy[Policy Index (PDFs)]
    Executor --> Domains[(Financial Domains)]
    FAISS -->|Contextual Matches| Executor
    Domains -->|Results| Executor
    Executor -->|Formatted Answer| User(UI)
```

------------------------------------------------------------------------

## 4. Core Components

  -----------------------------------------------------------------------
  Component                          Description
  ---------------------------------- ------------------------------------
  **Streamlit UI**                   Captures and sends user questions.

  **Planner**                        Uses GPT-4 or similar LLM to map
                                     intent → domain capabilities.

  **Executor**                       Runs structured API calls and
                                     retrieves contextual data.

  **FAISS Registry**                 Builds embeddings and vector
                                     indexes.

  **Temporal Registry**              Adds time-awareness to financial
                                     records.

  **Policy Index**                   Extracts text from PDF and builds
                                     vector retrieval space.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 5. Data Flow

``` mermaid
sequenceDiagram
    participant User
    participant Planner
    participant Executor
    participant FAISS
    participant Domains

    User->>Planner: "Did I buy anything related to Apple?"
    Planner->>Executor: Structured plan (domain=transactions, cap=semantic_search)
    Executor->>FAISS: Vector similarity search on transactions
    FAISS-->>Executor: Top K matches
    Executor->>User: Returns formatted result
```

------------------------------------------------------------------------

## 6. Planner

The **Planner** uses OpenAI GPT-4o-mini (configurable via `app.yaml`) to
interpret user questions into structured execution plans.\
Example:

``` json
{
  "plan": {
    "intent": "find_by_merchant",
    "calls": [
      {"domain_id": "transactions", "capability": "find_by_merchant", "args": {"merchant_query": "Apple"}}
    ]
  }
}
```

------------------------------------------------------------------------

## 7. Executor

The **Executor** runs the capabilities defined by the planner.\
It loads domain data (`transactions.json`, `payments.json`, etc.), and
executes deterministic or semantic logic.

Pseudocode:

``` python
if domain == "transactions":
    if capability == "semantic_search":
        results = query_index("transactions", query, embedder)
    elif capability == "find_by_merchant":
        results = txn_calc.find_by_merchant(txns, merchant_query)
```

------------------------------------------------------------------------

## 8. FAISS Index Engine

The FAISS layer is responsible for vector similarity search.\
During startup (`startup.py`), it indexes JSON and PDF files into vector
spaces.

``` mermaid
graph TD
    A[Raw Data Files] --> B[Embedder]
    B --> C[FAISS Index Builder]
    C --> D[Vector Index Store]
```

------------------------------------------------------------------------

## 9. Temporal Registry

The **Temporal Registry** (`temporal_registry.py`) builds time-aware
indexes for date filtering like "last month" or "recent transactions."\
It identifies date fields in JSON files (e.g., `transactionDateTime`,
`closingDateTime`) and attaches them to FAISS entries.

------------------------------------------------------------------------

## 10. Policy Index

Handles unstructured documents like **Apple Card Agreement**.\
`ensure_policy_index()` converts the PDF → text → embeddings for
retrieval.\
When users ask policy-related questions (e.g., "What is the late fee?"),
it queries this vector index.

------------------------------------------------------------------------

## 11. Context & Hint Packs

Hints are small YAML or Python-based prompt helpers
(`core/context/packs`).\
They provide structured patterns like:

    "if question mentions 'interest' and 'last' → intent=last_interest"

------------------------------------------------------------------------

## 12. Domain Calculators

Each financial domain contains deterministic logic modules:

  Domain            Calculator
  ----------------- -------------
  transactions      `txn_calc`
  payments          `pay_calc`
  statements        `stmt_calc`
  account_summary   `acct_calc`

------------------------------------------------------------------------

## 13. Semantic Search

Semantic search leverages FAISS embeddings for contextual retrieval.\
Example for transactions:

``` python
hits = query_index("transactions", "Apple purchases", embedder=embedder)
```

It then merges results by score and filters by `accountId` or `period`.

------------------------------------------------------------------------

## 14. Query Execution Flow

``` mermaid
flowchart TD
    A[User Query] --> B[Planner]
    B --> C[Structured Plan]
    C --> D[Executor]
    D --> E[Capability Dispatch]
    E --> F{Deterministic or Semantic?}
    F -->|Semantic| G[FAISS Query]
    F -->|Deterministic| H[Domain Calculator]
    G --> I[Results]
    H --> I[Results]
    I --> J[Formatted Output]
```

------------------------------------------------------------------------

## 15. Error Handling

  ----------------------------------------------------------------------------------------------
  Error                                  Description                  Resolution
  -------------------------------------- ---------------------------- --------------------------
  `Illegal header value b'Bearer '`      Empty API key                Ensure `OPENAI_API_KEY` is
                                                                      exported.

  `Unknown capability semantic_search`   Planner produced unsupported Add mapping in
                                         call                         `execute.py`.

  `No information found`                 No results returned          Rebuild FAISS indexes.
  ----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 16. Deployment

The system runs locally or via container.\
A `startup.py` build ensures indexes are pre-generated before Streamlit
loads.

Sample YAML:

``` yaml
indexes:
  dir: var/indexes
  rebuild_on_startup: true
```

------------------------------------------------------------------------

## 17. Extending the System

Add new domains under `domains/` with structure:

    /domains/<domain_name>/loader.py
    /domains/<domain_name>/calculator.py

Update `execute.py` to register new capabilities.

------------------------------------------------------------------------

## 18. Troubleshooting

  ------------------------------------------------------------------------
  Symptom             Possible Cause                 Solution
  ------------------- ------------------------------ ---------------------
  Planner hangs       Missing API key                Verify environment
                                                     variables

  Incorrect answers   Outdated FAISS index           Rebuild via
                                                     `startup.py`

  PDF not loading     Wrong file path                Check `policy.pdf`
                                                     path in `app.yaml`
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 19. Example Queries

  Example                                    Response Source
  ------------------------------------------ -------------------------------
  "What was my last transaction?"            transactions.last_transaction
  "Where did I spend the most?"              transactions.top_merchants
  "When was I last charged interest?"        statements.total_interest
  "Do I have any ACH payments?"              payments.semantic_search
  "What is the late fee per Apple policy?"   policy.index

------------------------------------------------------------------------

## 20. Appendix: Mermaid Diagrams

Developers can copy the Mermaid code blocks directly into GitHub or
Confluence to render interactive flowcharts.

``` mermaid
graph LR
    A[User Input] --> B[Planner]
    B --> C[Executor]
    C --> D[FAISS Search]
    D --> E[Results]
    E --> F[Streamlit Display]
```
