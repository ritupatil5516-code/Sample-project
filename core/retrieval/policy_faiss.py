from pathlib import Path
from typing import List, Dict, Any
import json, numpy as np, faiss, yaml
from pypdf import PdfReader
from core.embeddings.providers import build_embedder_from_config

ROOT = Path(".")
def _cfg_app() -> dict:
    p = ROOT / "config" / "app.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try: parts.append(page.extract_text() or "")
        except Exception: parts.append("")
    return "\n".join(parts)

def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if not text: raise RuntimeError("Policy text is empty after extraction; check the PDF file.")
    words = text.split()
    if not words: raise RuntimeError("Policy text contained no words; check the PDF file.")
    chunks = []; step = max(1, chunk_size - overlap); i=0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size])); i += step
    return chunks

def ensure_policy_index() -> None:
    cfg = _cfg_app()
    pol = cfg.get("policy") or {}
    if not pol.get("enabled", False): return
    store_dir = Path(pol.get("store_dir","var/policies")); store_dir.mkdir(parents=True, exist_ok=True)
    index_path = store_dir / "policy.index"
    meta_path  = store_dir / "policy_meta.json"
    if index_path.exists() and meta_path.exists(): return

    pdf_path = Path(pol.get("pdf_path","data/agreement/Apple-Card-Customer-Agreement.pdf"))
    if not pdf_path.exists():
        raise FileNotFoundError(f"Policy PDF not found at {pdf_path}")
    text = _extract_text(pdf_path)
    chunks = _chunk_text(text, 900, 150)
    embedder = build_embedder_from_config(cfg)
    model = (cfg.get("embeddings") or {}).get("openai_model","text-embedding-3-large")
    embs = embedder.embed(chunks, model=model)
    if not embs:
        raise RuntimeError("Embedding failed or returned empty for policy PDF")
    dim = len(embs[0]); index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs, dtype="float32"))
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps({i: chunks[i] for i in range(len(chunks))}, indent=2))

def query_policy(query: str, top_k: int = 3) -> Dict[str, Any]:
    cfg = _cfg_app()
    pol = cfg.get("policy") or {}
    if not pol.get("enabled", False):
        return {}
    store_dir = Path(pol.get("store_dir","var/policies"))
    index_path = store_dir / "policy.index"
    meta_path  = store_dir / "policy_meta.json"
    if not (index_path.exists() and meta_path.exists()):
        return {}
    meta = json.loads(meta_path.read_text())
    embedder = build_embedder_from_config(cfg)
    model = (cfg.get("embeddings") or {}).get("openai_model","text-embedding-3-large")
    q_emb = np.array([embedder.embed([query], model=model)[0]], dtype="float32")
    index = faiss.read_index(str(index_path))
    D, I = index.search(q_emb, top_k)
    hits = [str(i) for i in I[0] if str(i) in meta]
    snippets = [meta[h] for h in hits]
    return {"snippet": " ".join(snippets), "snippets": snippets, "ids": hits, "scores": [float(x) for x in D[0].tolist()], "citation": str(pol.get("pdf_path"))}
