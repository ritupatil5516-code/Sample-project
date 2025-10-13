
INTENT_EXAMPLES = [

    # 🧠 1. Direct Field Retrieval
    {
        "intent": "get_field_account_status",
        "calls": [
            {"op": "get_field", "domain": "account_summary", "args": {"key_path": "accountStatus"}}
        ]
    },
    {
        "intent": "get_field_available_credit",
        "calls": [
            {"op": "get_field", "domain": "account_summary", "args": {"key_path": "availableCreditAmount"}}
        ]
    },
    {
        "intent": "get_field_minimum_due",
        "calls": [
            {"op": "get_field", "domain": "account_summary", "args": {"key_path": "minimumDueAmount"}}
        ]
    },

    # 🕐 2. Find Latest (single record)
    {
        "intent": "latest_statement_close_date",
        "calls": [
            {"op": "find_latest", "domain": "statements", "args": {"ts_field": "closingDateTime", "value_path": "closingDateTime"}}
        ]
    },
    {
        "intent": "latest_interest_charged",
        "calls": [
            {"op": "find_latest", "domain": "statements", "args": {"ts_field": "closingDateTime", "value_path": "interestCharged"}}
        ]
    },
    {
        "intent": "latest_payment_amount",
        "calls": [
            {"op": "find_latest", "domain": "payments", "args": {"ts_field": "paymentPostedDateTime", "value_path": "amount"}}
        ]
    },
    {
        "intent": "last_posted_purchase",
        "calls": [
            {"op": "find_latest", "domain": "transactions", "args": {
                "ts_field": "postedDateTime",
                "value_path": "merchantName",
                "where": "[?transactionStatus=='POSTED' && contains(displayTransactionType,'PURCHASE') && amount > `0`]"
            }}
        ]
    },

    # 💰 3. Summation Queries
    {
        "intent": "total_spend_last_12m",
        "calls": [
            {"op": "sum_where", "domain": "transactions", "args": {
                "value_path": "amount",
                "where": "[?transactionStatus=='POSTED' && contains(displayTransactionType,'PURCHASE') && amount > `0`]"
            }}
        ]
    },
    {
        "intent": "total_payments_in_year",
        "calls": [
            {"op": "sum_where", "domain": "payments", "args": {
                "value_path": "amount",
                "where": "[?contains(paymentPostedDateTime, '2025-')]"
            }}
        ]
    },
    {
        "intent": "total_interest_all",
        "calls": [
            {"op": "sum_where", "domain": "statements", "args": {"value_path": "interestCharged"}}
        ]
    },

    # 🏆 4. Top-K Rankings
    {
        "intent": "top_merchants",
        "calls": [
            {"op": "topk_by_sum", "domain": "transactions", "args": {
                "group_key": "merchantName",
                "value_path": "amount",
                "where": "[?transactionStatus=='POSTED' && contains(displayTransactionType,'PURCHASE') && amount > `0`]",
                "k": 5
            }}
        ]
    },
    {
        "intent": "top_categories",
        "calls": [
            {"op": "topk_by_sum", "domain": "transactions", "args": {
                "group_key": "category",
                "value_path": "amount",
                "where": "[?transactionStatus=='POSTED' && amount > `0`]",
                "k": 5
            }}
        ]
    },

    # 📃 5. Filtered Listings
    {
        "intent": "transactions_from_merchant",
        "calls": [
            {"op": "list_where", "domain": "transactions", "args": {
                "where": "[?contains(merchantName, 'Apple')]"
            }}
        ]
    },
    {
        "intent": "transactions_above_threshold",
        "calls": [
            {"op": "list_where", "domain": "transactions", "args": {
                "where": "[?amount > `500` && transactionStatus=='POSTED']"
            }}
        ]
    },

    # 6. Multi-step
    {
      "intent": "last_purchase_details",
      "ops": [
        {"op":"find_latest","domain":"transactions","args":{
          "ts_field":"postedDateTime",
          "value_path":"merchantName",
          "where":"[?transactionStatus=='POSTED' && contains(displayTransactionType,'PURCHASE') && amount > `0`]"
        }},
        {"op":"find_latest","domain":"transactions","args":{
          "ts_field":"postedDateTime",
          "value_path":"amount",
          "where":"[?transactionStatus=='POSTED' && contains(displayTransactionType,'PURCHASE') && amount > `0`]"
        }}
      ],
      "must_produce": [],
      "risk_if_missing": []
    },

    # 7. Semantic retrieval
    {
      "intent": "semantic_travel",
      "ops": [
        {"op":"semantic_search","domain":"transactions","args":{
          "query":"travel purchases",
          "k":10
        }}
      ],
      "must_produce": [],
      "risk_if_missing": []
    },
    {
      "intent": "semantic_apple",
      "ops": [
        {"op":"semantic_search","domain":"transactions","args":{
          "query":"Apple related purchases",
          "k":10
        }}
      ],
      "must_produce": [],
      "risk_if_missing": []
    }
]