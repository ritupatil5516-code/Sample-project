import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

def load_transactions(path: Union[str, Path], mapping: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    return data if isinstance(data, list) else []