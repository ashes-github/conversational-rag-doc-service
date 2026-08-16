"""Session-scoped vector storage and document retrieval."""

import hashlib
from threading import RLock

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings

from backend.core.config import Settings
from backend.models.responses import IndexedDocument


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        self._top_k = settings.retrieval_top_k
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self._vectorstores: dict[str, Chroma] = {}
        self._documents: dict[str, list[IndexedDocument]] = {}
        self._lock = RLock()

    @staticmethod
    def _collection_name(session_id: str) -> str:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return f"session-{digest}"

    def _get_or_create_vectorstore(self, session_id: str) -> Chroma:
        vectorstore = self._vectorstores.get(session_id)
        if vectorstore is None:
            vectorstore = Chroma(
                collection_name=self._collection_name(session_id),
                embedding_function=self._embeddings,
            )
            self._vectorstores[session_id] = vectorstore
        return vectorstore

    def add_documents(
        self,
        session_id: str,
        filename: str,
        documents: list[Document],
    ) -> IndexedDocument:
        with self._lock:
            vectorstore = self._get_or_create_vectorstore(session_id)
            vectorstore.add_documents(documents)
            indexed = IndexedDocument(filename=filename, chunks=len(documents))
            self._documents.setdefault(session_id, []).append(indexed)
            return indexed

    def retrieve(self, query: str, config: RunnableConfig) -> list[Document]:
        session_id = str(config.get("configurable", {}).get("session_id", "")).strip()
        if not session_id:
            return []

        with self._lock:
            vectorstore = self._vectorstores.get(session_id)
            if vectorstore is None:
                return []
            return vectorstore.similarity_search(query, k=self._top_k)

    def list_documents(self, session_id: str) -> list[IndexedDocument]:
        with self._lock:
            return list(self._documents.get(session_id, []))
