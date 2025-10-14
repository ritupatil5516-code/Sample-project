# --- imports (version-safe) ---
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

try:
    # LC >= 0.2
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:  # older LC
    from langchain.schema import Document as LCDocument
    try:
        from langchain.retrievers.base import BaseRetriever
    except Exception:
        from langchain.retrievers import BaseRetriever
    try:
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun
    except Exception:
        CallbackManagerForRetrieverRun = None  # not used in older LC

try:
    # LC >= 0.2
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:
    # some older builds expose it directly
    from langchain.retrievers import EnsembleRetriever

from src.api.contextApp.retrieval.json_ingest import ensure_account_retriever
from src.api.contextApp.retrieval.knowledge_ingest import ensure_knowledge_retriever


# --- LLM factory (unchanged) ---
def _llm_from_cfg():
    # read your cfg/env the same way you already do
    # keep model/keys as you have them
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# --- LlamaIndex -> LangChain retriever shim ---
class _LlamaIndexToLangchainRetriever(BaseRetriever):
    """Wrap a LlamaIndex retriever so LC chains can use it."""

    li_retriever: object  # pydantic field

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[LCDocument]:
        # This is the ONLY place we ever call `.retrieve`:
        nodes = self.li_retriever.retrieve(query)
        docs: list[LCDocument] = []
        for n in nodes:
            text = getattr(n, "text", None) or (n.get_content() if hasattr(n, "get_content") else "")
            meta = dict(getattr(n, "metadata", {}) or {})
            docs.append(LCDocument(page_content=text or "", metadata=meta))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[LCDocument]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def _crc(llm, retriever: BaseRetriever, session_id: str):
    mem = ConversationBufferWindowMemory(
        k=10, return_messages=True, memory_key="chat_history"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=mem, return_source_documents=True
    )


# ------------------ PUBLIC APIS used by execute.py ------------------

def unified_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> dict:
    llm = _llm_from_cfg()

    # >>> THESE return LC *wrappers* around LlamaIndex retrievers
    r_acc = ensure_account_retriever(account_id=account_id, k=k)
    r_kn  = ensure_knowledge_retriever(k=k)

    retriever = EnsembleRetriever(retrievers=[r_acc, r_kn], weights=[0.7, 0.3])
    chain = _crc(llm, retriever=retriever, session_id=session_id)

    out = chain.invoke({"question": question})
    srcs = out.get("source_documents") or []
    sources = [{"source": d.metadata.get("source"), "snippet": (d.page_content or "")[:180]} for d in srcs[:5]]
    return {"answer": out.get("answer") or out.get("result"), "sources": sources}


def account_rag_answer(question: str, session_id: str, account_id: str, k: int = 6) -> dict:
    llm = _llm_from_cfg()
    r_acc = ensure_account_retriever(account_id=account_id, k=k)
    chain = _crc(llm, retriever=r_acc, session_id=session_id)
    out = chain.invoke({"question": question})
    return {"answer": out.get("answer") or out.get("result")}


def knowledge_rag_answer(question: str, session_id: str, k: int = 6) -> dict:
    llm = _llm_from_cfg()
    r_kn = ensure_knowledge_retriever(k=k)
    chain = _crc(llm, retriever=r_kn, session_id=session_id)
    out = chain.invoke({"question": question})
    return {"answer": out.get("answer") or out.get("result")}