import json
from pathlib import Path
from typing import List, Dict, Any, Union

def load_statements(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists(): return []
    try: data = json.loads(p.read_text())
    except Exception: return []
    if isinstance(data, list): return data
    if isinstance(data, dict): return [data]
    return []


# statements.py
def load(account_id: str, cfg: dict) -> list[dict]:
    rows = _read_json(cfg, account_id, "statements.json")
    for r in rows:
        # map vendor field names -> normalized names used by DSL
        if "interestAmount" in r and "interestCharged" not in r:
            r["interestCharged"] = r["interestAmount"]
        # ensure an ISO date key exists
        if "statementClose" in r and "closingDateTime" not in r:
            r["closingDateTime"] = r["statementClose"]
    return rows