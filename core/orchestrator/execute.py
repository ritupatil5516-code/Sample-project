import os
from typing import Dict, Any, List

from domains.transactions.loader import load_transactions
from domains.payments.loader import load_payments
from domains.statements.loader import load_statements
from domains.account_summary.loader import load_account_summary
from domains.transactions import calculator as txn_calc
from domains.payments import calculator as pay_calc
from domains.statements import calculator as stmt_calc
from domains.account_summary import calculator as acct_calc
from core.retrieval.policy_index import get_policy_snippet
# add these imports
from core.index.faiss_registry import query_index, Embedder
import yaml

def _stmt_key(s: dict) -> str:
    # prefer closingDateTime, else period, else empty
    return (s.get("closingDateTime") or s.get("period") or "")

def _pick_latest_stmt(stmts: list[dict], nonzero: bool) -> dict | None:
    if nonzero:
        cand = [s for s in stmts if (s.get("interestCharged") or 0) > 0]
    else:
        cand = stmts
    if not cand:
        return None
    return max(cand, key=_stmt_key)

def _latest_txn_key(t: dict) -> str:
    return (t.get("transactionDateTime") or t.get("postedDateTime") or t.get("date") or "")

def _fmt_money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"

def _latest_statement_period_from(stmts: List[dict]):
    periods = [s.get("period") for s in stmts if isinstance(s, dict) and s.get("period")]
    return max(periods) if periods else None

def _latest_statement_period_with_interest(stmts: List[dict]):
    periods = sorted({s.get("period") for s in stmts if s.get("period")}, reverse=True)
    for p in periods:
        for s in stmts:
            if s.get("period") == p:
                val = float(s.get("interestCharged") or s.get("interest_charged") or 0)
                if val > 0: return p
    return None

def _read_app_cfg(app_yaml_path: str) -> dict:
    try:
        with open(app_yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _build_embedder_from_cfg(cfg: dict) -> Embedder:
    """Builds the FAISS embedder using your embeddings config.
    (Supports OpenAI today; Qwen can be added later.)"""
    emb = (cfg.get("embeddings") or {})
    provider = (emb.get("provider") or "openai").lower()
    model = emb.get("openai_model") or emb.get("model") or "text-embedding-3-large"
    # OpenAI base/key via env; Embedder() reads OPENAI_API_KEY/OPENAI_API_BASE by default
    api_base = emb.get("openai_base_url") or emb.get("api_base") or None
    api_key_env = emb.get("openai_api_key_env") or "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env, "")
    return Embedder(provider=provider, model=model, api_key=api_key, api_base=api_base)

from datetime import datetime, timedelta, timezone

def _within_period(txn, period: str | None) -> bool:
    if not period:
        return True
    # Only implement LAST_12M for now (extend later if you like)
    if period.upper() == "LAST_12M":
        ts = (txn.get("transactionDateTime") or txn.get("postedDateTime") or txn.get("date"))
        if not ts:
            return False
        # parse ISO (very tolerant)
        try:
            # handle Z
            if ts.endswith("Z"):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(ts)
        except Exception:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        return dt >= cutoff
    return True

def execute_calls(calls: List[dict], config_paths: dict) -> Dict[str, Any]:
    txns = load_transactions("data/folder/transactions.json")
    pays = load_payments("data/folder/payments.json")
    stmts = load_statements("data/folder/statements.json")
    acct  = load_account_summary("data/folder/account_summary.json")
    cfg = _read_app_cfg(config_paths["app_yaml"])
    index_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    embedder = _build_embedder_from_cfg(cfg)


    results = {}
    latest_stmt_period = _latest_statement_period_from(stmts)
    intent = (config_paths or {}).get("intent")

    for call in calls:
        dom = call.get("domain_id"); cap = call.get("capability"); args = dict(call.get("args", {}))
        key = f"{dom}.{cap}"

        if dom == "transactions":
            if cap == "semantic_search":
                # Natural-language search over transaction embeddings (FAISS).
                q = (args.get("query") or "").strip()
                k = int(args.get("k", 5))
                if not q:
                    res = {"error": "query is required", "trace": {"k": k}}
                else:
                    hits = query_index(
                        domain="transactions",
                        query=q,
                        top_k=max(1, k),
                        index_dir=index_dir,
                        embedder=embedder,
                    )
                    # optional account_id filter (post-filter)
                    aid = args.get("account_id")
                    out = []
                    for h in hits:
                        payload = h.get("payload") or {}
                        if aid and payload.get("accountId") != aid:
                            continue
                        out.append({
                            "score": h.get("score"),
                            "text": h.get("text"),
                            "payload": payload,
                        })
                    res = {"hits": out, "trace": {"k": k, "query": q, "filtered_by_account": bool(aid)}}

            if cap == "list_over_threshold":
                thr = float(args.get("threshold", 0))
                res = txn_calc.list_over_threshold(txns, args.get("account_id"), thr, args.get("period"))
            elif cap == "last_transaction":
                # optional account filter
                aid = args.get("account_id")
                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                if not cand:
                    res = {"error": "No transactions", "trace": {"count": 0}}
                else:
                    last = max(cand, key=_latest_txn_key)
                    res = {"item": last, "trace": {"count": len(cand)}}


            elif cap == "top_merchants":

                aid = args.get("account_id")

                period = args.get("period")  # e.g., "LAST_12M"

                cand = [t for t in txns if (not aid or t.get("accountId") == aid)]

                # apply period filter

                cand = [t for t in cand if _within_period(t, period)]

                # purchases only (DEBIT or positive amount)

                rows = [t for t in cand if (t.get("transactionType") == "DEBIT" or (t.get("amount") or 0) > 0)]

                totals = {}

                for t in rows:
                    m = (t.get("merchantName") or "UNKNOWN").strip()

                    totals[m] = totals.get(m, 0.0) + float(t.get("amount") or 0.0)

                top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

                res = {

                    "top_merchants": [{"merchant": m, "total": v} for m, v in top],

                    "trace": {"count": len(rows), "period": period or "ALL"}

                }
            elif cap == "find_by_merchant":
                # args: merchant_query (required, case-insensitive contains), optional account_id, optional period
                q = (args.get("merchant_query") or "").strip().lower()
                if not q:
                    res = {"error": "merchant_query is required", "trace": {}}
                else:
                    aid = args.get("account_id")
                    period = args.get("period")
                    cand = [t for t in txns if (not aid or t.get("accountId") == aid)]
                    cand = [t for t in cand if _within_period(t, period)]
                    hits = []
                    for t in cand:
                        m = (t.get("merchantName") or "").lower()
                        if q in m:
                            hits.append(t)
                    # sort newest first
                    hits.sort(key=_latest_txn_key, reverse=True)
                    res = {"items": hits, "count": len(hits), "trace": {"merchant_query": q, "period": period or "ALL"}}

            elif cap == "spend_in_period":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))
            elif cap == "purchases_in_cycle":
                res = txn_calc.spend_in_period(txns, args.get("account_id"), args.get("period"))
            elif cap == "max_amount":
                res = txn_calc.max_amount(
                    txns,
                    args.get("account_id"),
                    args.get("period"),
                    args.get("period_start"),
                    args.get("period_end"),
                    args.get("category"),
                    int(args.get("top", 1)),
                )
            elif cap == "aggregate_by_category":
                res = txn_calc.aggregate_by_category(
                    txns, args.get("account_id"),
                    args.get("period"),
                    args.get("period_start"),
                    args.get("period_end"),
                )
            elif cap == "average_per_month":
                res = txn_calc.average_per_month(
                    txns,
                    args.get("account_id"),
                    args.get("period"),
                    args.get("period_start"),
                    args.get("period_end"),
                    bool(args.get("include_credits", False)),
                )
            elif cap == "compare_periods":
                res = txn_calc.compare_periods(txns, args.get("account_id"), args.get("period1"), args.get("period2"))
            else:
                res = {"error": f"Unknown capability {cap}"}

        elif dom == "payments":
            if cap == "last_payment":
                res = pay_calc.last_payment(pays, args.get("account_id"))
            elif cap == "total_credited_year":
                yr = args.get("year")
                res = pay_calc.total_credited_year(pays, args.get("account_id"), int(yr) if yr else None)
            elif cap == "payments_in_period":
                res = pay_calc.payments_in_period(pays, args.get("account_id"), args.get("period"))
            else:
                res = {"error": f"Unknown capability {cap}"}

        elif dom == "statements":

            # fill missing/NULL period
            nonzero = bool(args.get("nonzero"))
            if not args.get("period"):
                pick = _pick_latest_stmt(stmts, nonzero=nonzero)
                if pick:
                    args["period"] = pick.get("period")
                    args["_picked_close"] = pick.get("closingDateTime")
                    args["_picked_interest"] = pick.get("interestCharged")
                else:
                    args["period"] = None

            if not args.get("period"):
                if args.get("nonzero") or intent == "last_interest":
                    args["period"] = _latest_statement_period_with_interest(stmts) or latest_stmt_period
                else:
                    args["period"] = latest_stmt_period

            if cap == "total_interest":
                prd = args.get("period")
                if prd is None:
                    res = {"error": "No statements available", "trace": {"period": None}}
                else:
                    if args.get("_picked_interest") is not None:
                        amt = args["_picked_interest"]
                        res = {"interest_total": amt,
                               "trace": {"period": prd, "close_date": args.get("_picked_close"), "nonzero": nonzero}}
                    else:
                        # find exact period
                        row = next((s for s in stmts if s.get("period") == prd), None)
                        if not row:
                            res = {"error": "No statement for period", "trace": {"period": prd}}
                        else:
                            amt = row.get("interestCharged") or 0.0
                            res = {"interest_total": amt,
                                   "trace": {"period": prd, "close_date": row.get("closingDateTime"),
                                             "nonzero": nonzero}}
            elif cap == "interest_breakdown":
                res = stmt_calc.interest_breakdown(stmts, args.get("account_id"), args["period"])
            elif cap == "trailing_interest":
                res = stmt_calc.trailing_interest(stmts, args.get("account_id"), args["period"])
            else:
                res = {"error": f"Unknown capability {cap}"}

            if isinstance(res, dict):
                tr = res.setdefault("trace", {})
                if isinstance(tr, dict) and not tr.get("period") and args.get("period"):
                    tr["period"] = args["period"]

        elif dom == "account_summary":
            if cap == "current_balance":
                res = acct_calc.current_balance(acct)
            elif cap == "available_credit":
                res = acct_calc.available_credit(acct)
            else:
                res = {"error": f"Unknown capability {cap}"}

        elif dom == "policy":
            res = get_policy_snippet(cap)

        else:
            res = {"error": f"Unknown domain {dom}"}

        results[key] = res

    return results
