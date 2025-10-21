# core/retrieval/rag_chain.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# LlamaIndex (load persisted FAISS indexes)
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore

# Your process-wide singletons (LLM, memory, cfg)
from src.core.runtime import RUNTIME


# --------------------------- LlamaIndex helpers ---------------------------

def _li_index_from_dir(persist_dir: str):
    """Load a LlamaIndex VectorStoreIndex from a FAISS persist_dir. Returns None if missing."""
    p = Path(persist_dir)
    if not p.exists():
        return None
    try:
        vs = FaissVectorStore.from_persist_dir(str(p))
        sc = StorageContext.from_defaults(vector_store=vs, persist_dir=str(p))
        return load_index_from_storage(sc)
    except Exception as e:
        print(f"[RAG] load_index_from_storage failed @ {persist_dir}: {e}")
        return None


def _li_retrieve(persist_dir: str, query: str, k: int) -> List[Any]:
    """Retrieve top-k nodes from a LlamaIndex index. Returns [] if index missing."""
    idx = _li_index_from_dir(persist_dir)
    if idx is None:
        return []
    try:
        ret = idx.as_retriever(similarity_top_k=int(k))
        return ret.retrieve(query) or []
    except Exception as e:
        print(f"[RAG] retrieve failed @ {persist_dir}: {e}")
        return []


def _node_to_source(n: Any) -> Tuple[str, str]:
    """
    Convert a LlamaIndex Node (or NodeWithScore) to (source, snippet).
    Tries common attrs across LI versions.
    """
    node = getattr(n, "node", None) or n
    # content
    text = getattr(node, "text", None)
    if not text and hasattr(node, "get_content"):
        try:
            text = node.get_content()
        except Exception:
            text = ""
    if text is None:
        text = ""
    # metadata/path
    md = getattr(node, "metadata", {}) or {}
    src = (
        md.get("source")
        or md.get("file_path")
        or md.get("path")
        or md.get("doc_id")
        or "doc"
    )
    return str(src), str(text)


def _merge_nodes(acc_nodes: List[Any], kn_nodes: List[Any], k: int) -> List[Any]:
    """Simple interleave preferring account evidence, then knowledge."""
    out: List[Any] = []
    i = j = 0
    while len(out) < k and (i < len(acc_nodes) or j < len(kn_nodes)):
        if i < len(acc_nodes):
            out.append(acc_nodes[i]); i += 1
        if len(out) >= k: break
        if j < len(kn_nodes):
            out.append(kn_nodes[j]); j += 1
    return out[:k]


def _build_context_snippets(nodes: List[Any], max_chars: int = 8000) -> Tuple[str, List[Dict[str, str]]]:
    """
    Turn nodes into a single context block plus structured sources.
    We keep sources even if we truncate the context.
    """
    pieces: List[str] = []
    sources: List[Dict[str, str]] = []
    total = 0
    for n in nodes:
        src, text = _node_to_source(n)
        snippet = text.strip().replace("\n", " ").strip()
        sources.append({"source": src, "snippet": snippet[:220]})
        chunk = f"[{src}]\n{text.strip()}\n"
        if total + len(chunk) > max_chars:
            # stop adding to the prompt but still keep sources
            break
        pieces.append(chunk)
        total += len(chunk)
    return "\n".join(pieces).strip(), sources


def _synthesize_with_llm(question: str, context_block: str) -> str:
    """
    Call your runtime LLM with a strict instruction to answer
    ONLY from the provided context.
    """
    chat = RUNTIME.chat()
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using ONLY the provided context snippets.\n"
                "If the context does not contain the answer, say: \"I don't know from the provided context.\""
            ),
        },
        {
            "role": "system",
            "content": "Context:\n" + (context_block or "[no context]"),
        },
        {
            "role": "user",
            "content": question.strip(),
        },
    ]
    try:
        resp = chat.invoke(messages)  # LangChain ChatOpenAI style
        text = getattr(resp, "content", None) or ""
        if not text and isinstance(resp, dict):
            text = resp.get("content", "") or resp.get("text", "")
        return text or ""
    except Exception as e:
        return f"(rag_llm_error: {type(e).__name__}: {e})"


# --------------------------- Public entry points ---------------------------

def unified_rag_answer(
    question: str,
    session_id: str,
    account_id: Optional[str],
    k: int = 6,
) -> Dict[str, Any]:
    """
    Manual conversational RAG:
      - account index: var/indexes/accounts/{account_id}
      - knowledge index: var/indexes/knowledge
    Uses LlamaIndex just for retrieval, then calls your runtime LLM directly.
    """
    cfg = RUNTIME.cfg or {}
    Settings.embed_model = RUNTIME.embedding()  # use your runtime embedder

    idx_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    acc_dir = (Path(idx_dir) / "accounts" / str(account_id)).as_posix() if account_id else None
    kn_dir  = (Path(idx_dir) / "knowledge").as_posix()

    acc_nodes = _li_retrieve(acc_dir, question, k) if acc_dir else []
    kn_nodes  = _li_retrieve(kn_dir,  question, k)

    if not acc_nodes and not kn_nodes:
        return {"answer": "I don’t have any indexed data available yet to answer this.", "sources": [], "error": "no_retriever"}

    nodes = _merge_nodes(acc_nodes, kn_nodes, k=max(len(acc_nodes)+len(kn_nodes), k))
    context_block, sources = _build_context_snippets(nodes)
    answer = _synthesize_with_llm(question, context_block)
    return {"answer": answer, "sources": sources}


def account_rag_answer(
    question: str,
    session_id: str,
    account_id: str,
    k: int = 6,
) -> Dict[str, Any]:
    cfg = RUNTIME.cfg or {}
    Settings.embed_model = RUNTIME.embedding()
    idx_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    acc_dir = (Path(idx_dir) / "accounts" / str(account_id)).as_posix()

    nodes = _li_retrieve(acc_dir, question, k)
    if not nodes:
        return {"answer": "I don’t see an index for this account yet.", "sources": [], "error": "no_account_index"}

    ctx, sources = _build_context_snippets(nodes)
    return {"answer": _synthesize_with_llm(question, ctx), "sources": sources}


def knowledge_rag_answer(
    question: str,
    session_id: str,
    k: int = 6,
) -> Dict[str, Any]:
    cfg = RUNTIME.cfg or {}
    Settings.embed_model = RUNTIME.embedding()
    idx_dir = ((cfg.get("indexes") or {}).get("dir")) or "var/indexes"
    kn_dir = (Path(idx_dir) / "knowledge").as_posix()

    nodes = _li_retrieve(kn_dir, question, k)
    if not nodes:
        return {"answer": "Knowledge index not found yet.", "sources": [], "error": "no_knowledge_index"}

    ctx, sources = _build_context_snippets(nodes)
    return {"answer": _synthesize_with_llm(question, ctx), "sources": sources}