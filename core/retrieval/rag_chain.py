# core/retrieval/rag_chain.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os

# ----------------- LangChain imports (robust across versions) -----------------
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except Exception:
    # Fallback shims if needed
    class AIMessage:
        def __init__(self, content: str): self.content = content
    class HumanMessage:
        def __init__(self, content: str): self.content = content
    class SystemMessage:
        def __init__(self, content: str): self.content = content

# Chat model
try:
    from langchain_openai import ChatOpenAI
except Exception:
    # Older path
    from langchain.chat_models import ChatOpenAI  # type: ignore

# Memory (robust import + fallback)
try:
    from langchain.memory import ConversationBufferWindowMemory
except Exception:
    # Minimal drop-in replacement with the subset of features we use
    class ConversationBufferWindowMemory:  # type: ignore
        def __init__(self, k: int = 10, memory_key: str = "chat_history",
                     return_messages: bool = True, output_key: Optional[str] = None):
            self.k = k
            self.memory_key = memory_key
            self.return_messages = return_messages
            self.output_key = output_key
            self._msgs: List[Any] = []

            class _ChatMemory:
                def __init__(self, outer): self.outer = outer
                def add_user_message(self, text: str): outer._append(("human", text))
                def add_ai_message(self, text: str):   outer._append(("ai", text))
            self.chat_memory = _ChatMemory(self)

        def _append(self, item):
            self._msgs.append(item)
            self._msgs = self._msgs[-(2 * self.k):]

        def load_memory_variables(self, _) -> Dict[str, Any]:
            hist: List[Any] = []
            for role, text in self._msgs[-(2 * self.k):]:
                try:
                    if role == "human":
                        hist.append(HumanMessage(content=text))
                    else:
                        hist.append(AIMessage(content=text))
                except Exception:
                    hist.append((role, text))
            return {self.memory_key: hist}

# Your retrievers (must be LangChain-compatible retrievers)
from core.retrieval.json_ingest import ensure_account_retriever
from core.retrieval.knowledge_ingest import ensure_knowledge_retriever

# Per-session memory cache (avoid referencing class in type annotation for safety)
_MEMORY_BY_SESSION = {}  # type: Dict[str, ConversationBufferWindowMemory]


# -------------------------- helpers --------------------------

def _get_llm(cfg: Dict[str, Any]) -> ChatOpenAI:
    llm_cfg = (cfg.get("llm") or {})
    model = llm_cfg.get("model") or "gpt-4o-mini"
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    api_key = (llm_cfg.get("api_key") or
               os.getenv(llm_cfg.get("api_key_env", "") or "OPENAI_API_KEY", ""))

    if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
        api_base = "https://" + api_base

    # langchain_openai (new) vs older signature
    try:
        return ChatOpenAI(model=model, api_key=api_key, base_url=api_base, temperature=0)
    except TypeError:
        return ChatOpenAI(model_name=model, openai_api_key=api_key, openai_api_base=api_base, temperature=0)


def _get_memory(session_id: str, k: int = 10) -> ConversationBufferWindowMemory:
    mem = _MEMORY_BY_SESSION.get(session_id)
    if mem:
        return mem
    mem = ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    _MEMORY_BY_SESSION[session_id] = mem
    return mem


def _format_docs(docs) -> str:
    parts = []
    for d in docs or []:
        try:
            meta = d.metadata if hasattr(d, "metadata") else {}
            src = meta.get("source") or meta.get("path") or meta.get("file") or ""
            parts.append(f"[{src}]\n{getattr(d, 'page_content', str(d))}")
        except Exception:
            parts.append(getattr(d, "page_content", str(d)))
    return "\n\n---\n\n".join(parts)


def _pull_docs(retriever: Any, question: str) -> List[Any]:
    """Call retriever in a version-safe way."""
    if retriever is None:
        return []
    # Prefer .invoke for LCEL retrievers
    try:
        docs = retriever.invoke(question)
        if isinstance(docs, list):
            return docs
    except Exception:
        pass
    # Classic retriever API
    try:
        docs = retriever.get_relevant_documents(question)
        if isinstance(docs, list):
            return docs
    except Exception:
        pass
    # LlamaIndex adapter might expose .retrieve()
    try:
        out = retriever.retrieve(question)
        # Some adapters return objects with .node.text
        docs = []
        for item in out:
            text = getattr(item, "text", None)
            if text is None and hasattr(item, "node"):
                text = getattr(item.node, "text", None) or getattr(item.node, "get_content", lambda: "")()
            if text:
                class _Doc:
                    def __init__(self, page_content, metadata=None):
                        self.page_content = page_content
                        self.metadata = metadata or {}
                docs.append(_Doc(text, {}))
        return docs
    except Exception:
        pass
    return []


# ------------------------- public API --------------------------

def unified_rag_answer(question: str,
                       session_id: str,
                       account_id: Optional[str],
                       cfg: Dict[str, Any],
                       k: int = 6) -> Dict[str, Any]:
    """
    Simpler, version-agnostic conversational RAG:
      - Fetch docs from account + knowledge retrievers sequentially
      - Build a minimal chat prompt with memory and context
      - Call the LLM directly
    Returns {"answer": str, "sources": []}
    """
    # LLM and memory
    llm = _get_llm(cfg)
    mem = _get_memory(session_id, k=10)

    # Prepare chat history (LangChain message objects if available)
    chat_history = mem.load_memory_variables({}).get("chat_history") or []

    # Retrievers
    acc_ret = ensure_account_retriever(account_id)
    knw_ret = ensure_knowledge_retriever()

    # Retrieve docs (no RunnableParallel)
    acc_docs = _pull_docs(acc_ret, question)
    knw_docs = _pull_docs(knw_ret, question)
    context = _format_docs(acc_docs + knw_docs)

    # Compose prompt
    sys_a = SystemMessage(
        content=(
            "You are a precise banking copilot. Answer ONLY from the provided context. "
            "If the context is insufficient, say 'I don't know'. Be concise and factual."
        )
    )
    sys_ctx = SystemMessage(content=f"Context:\n{context}") if context else SystemMessage(content="Context:\n")
    user = HumanMessage(content=question)

    messages = [sys_a] + chat_history + [user, sys_ctx]

    # Invoke model
    try:
        resp = llm.invoke(messages)
        text = getattr(resp, "content", str(resp))
    except Exception as e:
        text = f"(RAG error: {type(e).__name__}: {e})"

    # Update memory
    try:
        mem.chat_memory.add_user_message(question)
        mem.chat_memory.add_ai_message(text)
    except Exception:
        pass

    return {"answer": text, "sources": []}