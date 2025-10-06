from typing import Dict, Any
def compose_answer(question: str, plan: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    if not results:
        return {"answer": "I couldn't find relevant data.", "citations": []}
    if "account_summary.current_balance" in results:
        r = results["account_summary.current_balance"]
        return {"answer": f"Your current balance is ${r.get('current_balance',0):,.2f} (as of {r.get('as_of_date')}).","citations": []}
    if "account_summary.available_credit" in results:
        r = results["account_summary.available_credit"]
        return {"answer": f"Available credit is ${r.get('available_credit',0):,.2f}; credit limit is ${r.get('credit_limit',0):,.2f}.","citations": []}
    if "statements.total_interest" in results:
        r = results["statements.total_interest"]; per = (r.get("trace") or {}).get("period")
        return {"answer": f"Interest charged in {per} was ${r.get('interest_total',0):,.2f}.","citations": []}
    return {"answer": f"Here is what I found:\n{results}", "citations": []}
