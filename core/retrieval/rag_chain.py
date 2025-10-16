# core/retrieval/rag_chain.py
from __future__ import annotations
import os, json, yaml, httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- FAISS registry (already in your project) ----
try:
    from core.index.faiss_registry import query_index, Embedder
except Exception as e:
    raise RuntimeError("Missing core.index.faiss_registry (need query_index, Embedder).") from e

# ---------- tiny in-process memory by session (last 10 turns) ----------
_MEMORY: Dict[str, List[Tuple[str, str]]] = {}  # session_id -> [(role, text)]

def _mem_get(session_id: str) -> List[Tuple[str, str]]:
    return _MEMORY.setdefault(session_id, [])

def _mem_add(session_id: str, role: str, text: str, keep: int = 10):
    hist = _MEMORY.setdefault(session_id, [])
    hist.append((role, text))
    # keep last N user/assistant messages
    if len(hist) > keep * 2:
        del hist[: len(hist) - keep * 2]

# ---------- config helpers ----------
def _read_app_cfg(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists(): return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _build_embedder_from_cfg(cfg: Dict[str, Any]) -> Embedder:
    emb = (cfg.get("embeddings") or {})
    provider = (emb.get("provider") or "openai").strip().lower()
    if provider == "qwen":
        api_base = (emb.get("qwen_base_url") or emb.get("api_base") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        model    = (emb.get("qwen_model") or emb.get("model") or "qwen3-embedding").strip()
        key_env  = (emb.get("qwen_api_key_env") or "QWEN_API_KEY").strip()
    else:
        api_base = (emb.get("openai_base_url") or emb.get("api_base") or "https://api.openai.com/v1").strip()
        model    = (emb.get("openai_model") or emb.get("model") or "text-embedding-3-large").strip()
        key_env  = (emb.get("openai_api_key_env") or "OPENAI_API_KEY").strip()
    api_key = (os.getenv(key_env) or "").strip()
    return Embedder(provider=provider, model=model, api_key=api_key, api_base=api_base)

# ---------- tiny OpenAI-compatible LLM client (httpx) ----------
class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def invoke(self, messages: List[Dict[str, Any]], model: Optional[str] = None, temperature: float = 0.0) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model or self.model, "messages": messages, "temperature": float(temperature)}
        with httpx.Client(timeout=40.0) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                # print server error for quick diagnosis
                print("[LLM ERROR]", r.status_code, r.text)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

def _get_llm(cfg: Dict[str, Any]) -> _LLMClient:
    llm_cfg = (cfg.get("llm") or {})
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    if api_base and not api_base.startswith("http"): api_base = "https://" + api_base
    api_key  = (llm_cfg.get("api_key") or os.getenv(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "") or llm_cfg.get("key") or "").strip()
    model    = (llm_cfg.get("model") or "gpt-4o-mini").strip()
    if not api_key:
        raise RuntimeError("Missing LLM API key. Set llm.api_key or export env named by llm.api_key_env.")
    return _LLMClient(api_base=api_base, api_key=api_key, model=model)

# ---------- retrieval helpers using your FAISS indexes ----------
def _faiss_hits(
    domain: str,
    query: str,
    embedder: Embedder,
    index_dir: str,
    top_k: int = 6,
    account_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a normalized list of hits: {"text", "payload", "score"}
    Filters by account_id if payload.accountId matches in the index payload.
    """
    try:
        hits = query_index(domain=domain, query=query, top_k=int(top_k), index_dir=index_dir, embedder=embedder)
    except Exception as e:
        print(f"[RAG] query_index failed for {domain}: {e}")
        return []
    out = []
    for h in hits or []:
        payload = h.get("payload") or {}
        if account_id and payload.get("accountId") not in (account_id, str(account_id)):
            # keep only matching account if available
            continue
        out.append({"text": h.get("text") or "", "payload": payload, "score": float(h.get("score", 0.0))})
    return out

def _format_docs(docs: List[Dict[str, Any]], max_len: int = 1600) -> str:
    """
    Builds a compact context block that keeps within a rough token budget.
    One line per doc: [source] snippet
    """
    lines: List[str] = []
    total = 0
    for d in docs:
        p = d.get("payload") or {}
        src = p.get("source") or p.get("path") or p.get("file") or p.get("domain") or "doc"
        snippet = (d.get("text") or "").strip().replace("\n", " ")
        if not snippet: continue
        line = f"[{src}] {snippet}"
        if total + len(line) > max_len:
            break
        lines.append(line); total += len(line)
    return "\n".join(lines)

def _collect_sources(docs: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, str]]:
    out = []
    for d in docs[:limit]:
        p = d.get("payload") or {}
        out.append({
            "source": p.get("source") or p.get("path") or p.get("file") or p.get("domain") or "",
            "snippet": (d.get("text") or "")[:200]
        })
    return out

# ---------- public API ----------
def unified_rag_answer(
    question: str,
    session_id: str,
    account_id: Optional[str],
    cfg: Dict[str, Any],
    k: int = 6,
) -> Dict[str, Any]:
    """
    Single entry point used by execute.py when strategy=='rag' (or fallback).
    - Retrieves from account JSON indexes (transactions/payments/statements)
      and from knowledge index (handbook, policy/agreement).
    - Uses in-memory chat history (last ~10 user/assistant turns).
    - Calls your OpenAI-compatible LLM with the assembled context.
    Returns: {"answer": str, "sources": [{source, snippet}, ...]}
    """
    try:
        llm = _get_llm(cfg)
        embedder = _build_embedder_from_cfg(cfg)
        idx_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"

        # ---- 1) Retrieval (FAISS) ----
        # Account JSON domains
        acc_docs = []
        for dom in ("transactions", "payments", "statements"):
            acc_docs += _faiss_hits(domain=dom, query=question, embedder=embedder, index_dir=idx_dir, top_k=k, account_id=account_id)

        # Knowledge (handbook + agreement) – assume you built a 'knowledge' index
        kn_docs = _faiss_hits(domain="knowledge", query=question, embedder=embedder, index_dir=idx_dir, top_k=k, account_id=None)

        docs = sorted(acc_docs + kn_docs, key=lambda d: d.get("score", 0.0), reverse=True)[: max(3, k)]
        context_block = _format_docs(docs)
        sources = _collect_sources(docs)

        # ---- 2) Build messages with short-term memory ----
        hist = _mem_get(session_id)
        msg_hist: List[Dict[str, str]] = []
        for role, text in hist[-20:]:
            if not text: continue
            if role in ("human", "user"):
                msg_hist.append({"role": "user", "content": text})
            else:
                msg_hist.append({"role": "assistant", "content": text})

        sys = {"role": "system", "content":
               "You are a precise banking copilot. Answer ONLY using the provided context. "
               "If the context is insufficient, say \"I don't know\". Be concise and factual."}
        ctx = {"role": "system", "content": f"Context:\n{context_block}"} if context_block else {"role":"system","content":"Context: (no supporting passages found)"}
        user = {"role": "user", "content": question}

        messages = [sys] + msg_hist + [ctx, user]

        # ---- 3) LLM call ----
        answer = llm.invoke(messages, model=(cfg.get("llm") or {}).get("model"), temperature=0)

        # ---- 4) Update memory ----
        _mem_add(session_id, "human", question)
        _mem_add(session_id, "ai", answer)

        return {"answer": answer, "sources": sources}

    except Exception as e:
        return {"answer": "I hit an error running RAG.", "sources": [], "error": f"{type(e).__name__}: {e}"}