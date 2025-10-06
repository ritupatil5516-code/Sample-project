import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

def load_account_summary(path: Union[str, Path], mapping: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        return data[0]
    return {}