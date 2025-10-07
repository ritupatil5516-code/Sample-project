from pathlib import Path
from typing import List, Dict, Any
import json, numpy as np, faiss
from core.embeddings.providers import build_embedder_from_config
import yaml

INDEX_DIR = Path("var/indexes"); INDEX_DIR.mkdir(parents=True, exist_ok=True)

class DomainIndex:
    def __init__(self, domain: str, cfg: dict):
        self.domain = domain
        self.cfg = cfg
        self.index_path = INDEX_DIR / f"{domain}.index"
        self.meta_path  = INDEX_DIR / f"{domain}_meta.json"
        self.index = None; self.meta = {}

    def build(self, records: List[Dict[str, Any]], text_field: str = "text"):
        texts = [r.get(text_field) for r in records if r.get(text_field)]
        if not texts: return
        embedder = build_embedder_from_config(self.cfg)
        model = (self.cfg.get("embeddings") or {}).get("openai_model","text-embedding-3-large")
        embs = embedder.embed(texts, model=model)
        dim = len(embs[0]); idx = faiss.IndexFlatL2(dim)
        idx.add(np.array(embs, dtype="float32"))
        faiss.write_index(idx, str(self.index_path))
        self.meta = {str(i): texts[i] for i in range(len(texts))}
        self.meta_path.write_text(json.dumps(self.meta, indent=2))
        self.index = idx

    def load(self):
        if self.index_path.exists(): self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists(): self.meta = json.loads(self.meta_path.read_text())
        return self

    def ensure(self, records: List[Dict[str, Any]], text_field: str = "text"):
        if not (self.index_path.exists() and self.meta_path.exists()):
            self.build(records, text_field=text_field)
        else:
            self.load()
        return self
