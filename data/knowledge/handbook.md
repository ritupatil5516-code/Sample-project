# Copilot Handbook

This handbook contains human-readable notes for agents and a **machine-readable rules block** the planner can parse to create deterministic plans for common patterns (e.g., “last”, “latest”, “recent”, “most”).

## How to extend
Add new routes in the `planner-rules` block below. The planner tries these rules **before** asking the LLM. Each rule lists the tokens that must appear in the question (via `synonyms`) and the single tool call to make.

---

## Planner Rules

```planner-rules
version: 1

# Word families the planner will look for in the question text.
synonyms:
  recency: ["last", "latest", "recent", "most recent", "recently"]
  interest: ["interest", "finance charge", "interest charged", "interest fee"]
  spend: ["spend", "spent", "purchases", "charges", "debits"]
  most: ["most", "top", "largest", "biggest", "highest"]
  balance: ["balance", "current balance"]
  status: ["account status", "status"]

# Declarative routes. First match wins (ordering matters).
routes:
  # Latest statement with non-zero interest
  - name: last_interest_charge
    must: ["recency", "interest"]
    call:
      domain_id: statements
      capability: find_latest
      args:
        field: closingDateTime
        where:
          interestCharged: {">": 0}

  # (Fallback) Latest interest as a transaction
  - name: last_interest_transaction
    must: ["recency", "interest"]
    call:
      domain_id: transactions
      capability: find_latest
      args:
        field: postedDateTime
        where:
          category: "INTEREST"

  # “Where did I spend the most …” → topK by merchant over last 12 months
  - name: spend_most_last_12m
    must: ["spend", "most"]
    call:
      domain_id: transactions
      capability: topk_by_sum
      args:
        group_key: merchantName
        value_path: amount
        where:
          transactionType: "DEBIT"
          period: "LAST_12M"
        k: 5

  # Latest/last transaction (generic recency without other hints)
  - name: latest_transaction
    must: ["recency"]
    call:
      domain_id: transactions
      capability: last_transaction
      args: {}

  # Simple account facts
  - name: get_current_balance
    must: ["balance"]
    call:
      domain_id: account_summary
      capability: get_field
      args:
        field: currentBalance

  - name: get_account_status
    must: ["status"]
    call:
      domain_id: account_summary
      capability: get_field
      args:
        field: status