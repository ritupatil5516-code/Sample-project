from pathlib import Path
from typing import List, Dict, Any, Optional
import json, numpy as np, faiss, yaml
from pypdf import PdfReader
from core.index.embedder import embed_texts

ROOT = Path(".")
POLICY_DIR = ROOT / "var" / "policies"; POLICY_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = POLICY_DIR / "policy.index"
META_PATH  = POLICY_DIR / "policy_meta.json"
DEFAULT_PDF_PATH = ROOT / "data" / "agreement" / "Agreement.pdf"

def _cfg_app() -> dict:
    p = ROOT / "config" / "app.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def _pdf_path_from_cfg() -> Path:
    cfg = _cfg_app()
    p = (cfg.get("policy") or {}).get("pdf_path")
    if p:
        pp = Path(p); return pp if pp.is_absolute() else ROOT / p
    return DEFAULT_PDF_PATH

def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try: parts.append(page.extract_text() or "")
        except Exception: parts.append("")
    return "\n".join(parts)

def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if not text: raise RuntimeError("Policy text is empty after extraction; check the PDF file.")
    words = text.split(); 
    if not words: raise RuntimeError("Policy text contained no words; check the PDF file.")
    chunks = []; step = max(1, chunk_size - overlap); i=0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size])); i += step
    return chunks

def load_pdf_chunks(pdf_path: Path) -> List[str]:
    return _chunk_text(_extract_text(pdf_path), 900, 150)

def ensure_policy_index() -> None:
    if INDEX_PATH.exists() and META_PATH.exists(): return
    pdf_path = _pdf_path_from_cfg()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Policy PDF not found at {pdf_path}")
    chunks = load_pdf_chunks(pdf_path)  # may raise if empty
    embs = embed_texts(chunks)
    if not embs:
        raise RuntimeError("Embedding failed or returned empty for policy PDF")
    dim = len(embs[0]); index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs, dtype="float32")); 
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({i: chunks[i] for i in range(len(chunks))}, indent=2))

def query_policy(query: str, store_dir: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    if not (INDEX_PATH.exists() and META_PATH.exists()):
        return {}
    index = faiss.read_index(str(INDEX_PATH)); meta = json.loads(META_PATH.read_text())
    q_emb = np.array([embed_texts([query])[0]], dtype="float32")
    D, I = index.search(q_emb, top_k)
    hits = [str(i) for i in I[0] if str(i) in meta]
    snippets = [meta[h] for h in hits]
    return {"snippet": " ".join(snippets), "snippets": snippets, "ids": hits, "scores": [float(x) for x in D[0].tolist()], "citation": str(_pdf_path_from_cfg())}
