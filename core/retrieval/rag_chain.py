# core/retrieval/rag_chain.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import os

# --- LangChain v0.2+ imports (with graceful fallbacks) ---
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # older versions
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.prompts import MessagesPlaceholder  # might still exist
    from langchain.output_parsers import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chat_models import ChatOpenAI  # older
    from langchain.embeddings import OpenAIEmbeddings

# If you adapt LlamaIndex retrievers to LangChain, import your ensure_* functions:
# (Assuming you already implemented these to return LangChain-compatible Retrievers)
from core.retrieval.json_ingest import ensure_account_retriever
from core.retrieval.knowledge_ingest import ensure_knowledge_retriever

# Simple in-process memory store keyed by session_id.
# For multi-process use Redis or a DB-backed chat message store.
_MEMORY_BY_SESSION: Dict[str, ConversationBufferWindowMemory] = {}


def _get_llm(cfg: Dict[str, Any]) -> ChatOpenAI:
    llm_cfg = (cfg.get("llm") or {})
    model = llm_cfg.get("model") or "gpt-4o-mini"
    api_base = (llm_cfg.get("api_base") or llm_cfg.get("base_url") or "https://api.openai.com/v1").strip()
    api_key = (llm_cfg.get("api_key") or
               os.getenv(llm_cfg.get("api_key_env", "") or "OPENAI_API_KEY", ""))

    if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
        api_base = "https://" + api_base

    # ChatOpenAI (langchain_openai) supports base_url param name; older uses openai_api_base
    try:
        return ChatOpenAI(model=model, api_key=api_key, base_url=api_base, temperature=0)
    except TypeError:
        # fallback older signature
        return ChatOpenAI(model_name=model, openai_api_key=api_key, openai_api_base=api_base, temperature=0)


def _format_docs(docs) -> str:
    parts = []
    for d in docs or []:
        try:
            meta = d.metadata if hasattr(d, "metadata") else {}
            src = meta.get("source") or meta.get("path") or meta.get("file") or ""
            parts.append(f"[{src}]\n{d.page_content}")
        except Exception:
            parts.append(getattr(d, "page_content", str(d)))
    return "\n\n---\n\n".join(parts)


def _get_memory(session_id: str, k: int = 10) -> ConversationBufferWindowMemory:
    mem = _MEMORY_BY_SESSION.get(session_id)
    if mem:
        return mem
    mem = ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",  # this name is used in the prompt below
        return_messages=True,
        output_key="answer",  # helpful if chain returns dict
    )
    _MEMORY_BY_SESSION[session_id] = mem
    return mem


def build_unified_rag_chain(cfg: Dict[str, Any],
                            account_id: Optional[str]) -> Any:
    """
    Build a conversational RAG chain that pulls from:
      - account retriever (JSON-derived vectors) for this account_id
      - knowledge retriever (handbook + agreement)
    Then merges results and passes to the LLM with memory.
    """
    llm = _get_llm(cfg)

    # Retrievers (LangChain-compatible)
    acc_ret = ensure_account_retriever(account_id)
    knw_ret = ensure_knowledge_retriever()

    # Merge two retrievers: retrieve from both, then concat and de-dup
    # RunnableParallel lets us run both in parallel in LCEL.
    parallel = RunnableParallel(acc=acc_ret, knw=knw_ret)

    def _merge_retrieval(inputs: Dict[str, Any]):
        """LCEL function to combine results from both retrievers."""
        # inputs == {"question": "..."} from RunnablePassthrough
        q = inputs.get("question") or inputs.get("input") or ""
        fetched = parallel.invoke(q)
        acc_docs = fetched.get("acc") or []
        knw_docs = fetched.get("knw") or []
        # simple concat; could add scoring/weights here
        combined = acc_docs + knw_docs
        # (optional) truncate to top-k here if necessary; retrievers usually handle k
        return {"context": _format_docs(combined), "question": q}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise banking copilot. "
         "Answer ONLY from the provided context. "
         "If the context is insufficient, say 'I don't know'. "
         "Be concise and factual."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("system", "Context:\n{context}")
    ])

    # LCEL: question passthrough -> merged retrieval -> prompt -> llm -> text
    chain = (
        RunnablePassthrough()
        | _merge_retrieval
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def _safe_invoke(chain: Any,
                 question: str,
                 memory: ConversationBufferWindowMemory,
                 chat_history: Optional[List] = None) -> str:
    """
    Invoke chain defensively across LangChain variants / input schemas.
    """
    # coerce chat_history for prompt (MessagesPlaceholder wants LC messages)
    if chat_history is None:
        chat_history = memory.load_memory_variables({}).get("chat_history") or []
    # The LCEL chain takes a dict with "question" and we inject chat_history via config.
    inputs = {"question": question}
    try:
        # Try standard .invoke with prompt vars in inputs and history in "configurable" metadata
        return chain.invoke(
            inputs,
            config={"configurable": {"chat_history": chat_history}}
        )
    except Exception as e1:
        # Try packing history directly into inputs
        try:
            return chain.invoke({"question": question, "chat_history": chat_history})
        except Exception as e2:
            # Try alternate key "input"
            try:
                return chain.invoke({"input": question, "chat_history": chat_history})
            except Exception:
                # Bubble the first error; it’s the closest to the true cause
                raise e1


def unified_rag_answer(question: str,
                       session_id: str,
                       account_id: Optional[str],
                       cfg: Dict[str, Any],
                       k: int = 6) -> Dict[str, Any]:
    """
    Public entry for executor: builds chain, invokes safely, records memory,
    and normalizes output shape.
    Returns: {"answer": str, "sources": []}
    """
    # Build/obtain chain
    chain = build_unified_rag_chain(cfg, account_id=account_id)

    # Memory
    mem = _get_memory(session_id, k=10)
    # Save the user question to memory first (some memory backends append after)
    mem.chat_memory.add_user_message(question)

    # Run
    text = _safe_invoke(chain, question, memory=mem)

    # Save assistant reply into memory
    mem.chat_memory.add_ai_message(text)

    # Normalize shape expected by compose/execute
    return {
        "answer": text,
        # If you need sources, modify _merge_retrieval to also return doc metadata.
        "sources": []
    }