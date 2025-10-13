# domains/transactions/loader.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _customer_data_root() -> Path:
    """
    Where per-account JSON folders live.
    Order of precedence:
      1) env CUSTOMERS_DATA_ROOT (or CUSTOMER_DATA_ROOT)
      2) src/api/contextApp/data/customer_data
      3) data/customer_data
    """
    env = os.getenv("CUSTOMERS_DATA_ROOT") or os.getenv("CUSTOMER_DATA_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p

    for rel in (
        "src/api/contextApp/data/customer_data",
        "data/customer_data",
    ):
        p = Path(rel)
        if p.exists():
            return p

    # fallback (may not exist yet; callers handle missing files gracefully)
    return Path("data/customer_data")


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    """
    Reads a JSON file that may be:
      - a list[dict]
      - a dict with key 'transactions' -> list
      - a single dict (wrap to list)
    Returns [] if file missing or invalid.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("transactions"), list):
            return [x for x in data["transactions"] if isinstance(x, dict)]
        return [data]
    return []


def load_transactions(account_id: str) -> List[Dict[str, Any]]:
    """
    Load transactions for a given account_id from:
        <DATA_ROOT>/<account_id>/transactions.json

    - Ensures a list[dict] is returned
    - If 'accountId' is missing in a row, fills it with the provided account_id
    """
    base = _customer_data_root()
    path = base / account_id / "transactions.json"
    rows = _read_json_list(path)

    # normalize accountId
    for r in rows:
        r.setdefault("accountId", account_id)

    return rows