import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

def load_statements(path: Union[str, Path], mapping: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []