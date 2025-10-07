# Credit Card Copilot Knowledge Handbook

This handbook is embedded as a semantic index to help the LLM reason about
credit card transactions, interest rules, payments, and balance calculations.
It complements the Apple Card Customer Agreement.

---

## 🔹 1. Core Credit Card Concepts

### Current Balance
The current balance represents the total amount owed on your credit card,
including purchases, fees, and interest accrued up to the statement closing date.
It excludes any pending transactions not yet posted.

### Available Credit
Available credit = Credit Limit − Current Balance.
This amount is the spending power left on the card before reaching the limit.

### Minimum Due
The smallest amount you must pay by the due date to keep the account in good standing.
Failing to pay it may incur late fees or affect your credit score.

---

## 🔹 2. Statements and Interest

### Statement Period
A statement covers a billing cycle (typically 30 days).  
It lists all posted transactions, payments, credits, and interest charges.

### Interest Calculation
Interest (finance charge) is calculated on the average daily balance for the billing period:
> **Interest = (Average Daily Balance × Daily Periodic Rate) × Number of Days in Cycle**

The **Daily Periodic Rate (DPR)** = APR ÷ 365.  
Different categories (purchases, cash advances, balance transfers) may have separate APRs.

### Trailing Interest
Trailing interest (or residual interest) occurs when you pay the full statement balance *after* the due date.
Interest continues to accrue until the payment posts.

### Grace Period
If you pay the full balance by the due date, no interest is charged on new purchases.
If you carry a balance, new purchases accrue interest immediately.

---

## 🔹 3. Payments

### Last Payment
Represents the most recent payment recorded, with its amount, date, and funding source.
Payments reduce the current balance and may restore available credit.

### Late or Partial Payment
If the payment is below the minimum due or received after the due date,
interest and late fees may apply.

### Total Payments in Year
Sum of all payments recorded within a given calendar year.  
Used for year-end summaries and reporting.

---

## 🔹 4. Transactions

### Purchases
Purchases are all debit transactions that reduce available credit.
Each transaction has:
- `transactionDateTime`
- `merchantName`
- `amount`
- `transactionType` (e.g., DEBIT)
- `cardType` (e.g., CREDIT)

### Last Transaction
The most recent transaction sorted by `transactionDateTime`.
This is typically the latest posted purchase.

### Spend in Period
Total purchase amount in a specified month or billing cycle.

### High-Value Transactions
Transactions over a certain threshold (e.g., $500) may be considered large purchases.

### Merchant Analysis
Grouping transactions by merchant helps identify where the user spends the most.

---

## 🔹 5. Interest Scenarios

### Example 1: Paid in Full
- Balance: $1,000
- Payment: $1,000 before due date  
→ **No interest charged.**

### Example 2: Partial Payment
- Balance: $1,000
- Payment: $500  
→ Interest applies to remaining $500 until it’s paid.

### Example 3: Late Payment
- Balance: $1,000
- Payment: $1,000 after due date  
→ Interest charged for the days between due date and payment posting.

---

## 🔹 6. Dispute or Refund

### Credit or Return
Credits and refunds (e.g., returned items) are treated as negative purchases
and reduce the outstanding balance.

### Dispute Transaction
If a transaction is disputed, it may temporarily not accrue interest until resolved.

---

## 🔹 7. Policy and Agreement Highlights (Apple Card)

> Extracted and summarized for reference. Full details in Apple Card Customer Agreement (PDF).

- No fees: Apple Card charges no annual or late fees, but interest still applies on unpaid balances.
- Daily Cash: Cash rewards are credited daily to your Apple Cash card.
- Variable APR: Based on creditworthiness and the Prime Rate.
- Payments can be made anytime through the Wallet app.
- Statement availability: Monthly, typically closing near the same calendar day each month.
- Grace period: Pay in full by the due date to avoid interest.
- Privacy: Apple and Goldman Sachs do not share transaction details for marketing.

---

## 🔹 8. Common User Questions

| Question | Likely Domain | Description |
|-----------|----------------|-------------|
| “What is my current balance?” | account_summary | Returns latest balance and as-of date |
| “When was I last charged interest?” | statements | Finds the most recent non-zero interest charge |
| “What was my last transaction?” | transactions | Retrieves the latest transaction by date |
| “How much did I spend in April?” | transactions | Sum of purchases in April |
| “What is my average purchase per month?” | transactions | Groups spend by period and averages |
| “Why was I charged interest?” | policy + statements | Combines billing and policy context |
| “When is my next payment due?” | account_summary | Uses paymentDueDate from account summary |
| “What is trailing interest?” | handbook + statements | Definition plus last statement data |

---

## 🔹 9. Data Mapping (for reference)

| Domain | Key Fields |
|---------|-------------|
| transactions | transactionId, merchantName, amount, transactionDateTime |
| payments | paymentId, amount, paymentDateTime |
| statements | statementId, period, interestCharged, totalPayments |
| account_summary | currentBalance, availableCredit, paymentDueDate |

---

## 🔹 10. Example Interpretations

- **"Last interest"** → latest non-zero interestCharged in `statements.json`.
- **"Recent transaction"** → max(`transactionDateTime`) in `transactions.json`.
- **"Average monthly spend"** → group by `period` (YYYY-MM), compute mean.
- **"Compare March vs April"** → list period totals side by side.

---

_End of handbook._