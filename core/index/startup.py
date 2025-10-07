from pathlib import Path
import yaml
from core.index.domain_faiss import DomainIndex
from core.retrieval.policy_faiss import ensure_policy_index
from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary

def _cfg():
    return yaml.safe_load(Path("config/app.yaml").read_text())

def _rows_to_text(rows, domain: str):
    chunks = []
    if domain == "transactions":
        for r in rows:
            chunks.append(f"{r.get('transactionDateTime') or r.get('transaction_date_time','')} | {r.get('merchantName') or r.get('merchant_name','')} | {r.get('transactionType') or r.get('transaction_type','')} | {r.get('amount','')}")
    elif domain == "payments":
        for r in rows:
            chunks.append(f"{r.get('paymentDateTime') or r.get('payment_date_time','')} | amount {r.get('amount','')} | posted {r.get('paymentPostedDateTime') or r.get('payment_posted_date_time','')}")
    elif domain == "statements":
        for r in rows:
            chunks.append(f"period {r.get('period','')} | interest {r.get('interestCharged') or r.get('interest_charged','')} | close {r.get('closingDateTime') or r.get('closing_date_time','')}")
    elif domain == "account_summary":
        if rows and isinstance(rows[0], dict):
            r = rows[0]
            chunks.append(f"balance {r.get('currentBalance') or r.get('current_balance','')} | available {r.get('availableCredit') or r.get('available_credit','')} | limit {r.get('creditLimit') or r.get('credit_limit','')}")
    else:
        for r in rows:
            chunks.append(str(r))
    return [{"text": t} for t in chunks if t]

def build_on_startup():
    cfg = _cfg()
    if not (cfg.get('index', {}).get('build_on_startup', True)): return

    txns = load_transactions('data/folder/transactions.json')
    pays = load_payments('data/folder/payments.json')
    stmts = load_statements('data/folder/statements.json')
    acct  = load_account_summary('data/folder/account_summary.json')
    acct_rows = [acct] if acct else []

    if txns: DomainIndex('transactions', cfg).ensure(_rows_to_text(txns, 'transactions'))
    if pays: DomainIndex('payments', cfg).ensure(_rows_to_text(pays, 'payments'))
    if stmts: DomainIndex('statements', cfg).ensure(_rows_to_text(stmts, 'statements'))
    if acct_rows: DomainIndex('account_summary', cfg).ensure(_rows_to_text(acct_rows, 'account_summary'))

    ensure_policy_index()
