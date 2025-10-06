from typing import Dict

def current_balance(acct: Dict):
    return {
        "current_balance": float(acct.get("currentBalance") or acct.get("current_balance") or 0),
        "as_of_date": acct.get("billingCycleCloseDateTime") or acct.get("billing_cycle_close_date_time"),
    }

def available_credit(acct: Dict):
    return {
        "available_credit": float(acct.get("availableCredit") or acct.get("available_credit") or 0),
        "credit_limit": float(acct.get("creditLimit") or acct.get("credit_limit") or 0),
        "as_of_date": acct.get("billingCycleCloseDateTime") or acct.get("billing_cycle_close_date_time"),
    }