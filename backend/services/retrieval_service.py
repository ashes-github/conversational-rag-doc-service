"""Session-scoped vector storage and document retrieval."""

import hashlib
from collections.abc import Callable, Sequence
from threading import RLock
from time import perf_counter
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings

from backend.core.config import Settings
from backend.models.responses import IndexedDocument
from backend.services.observability_service import ObservabilityService


class Reranker(Protocol):
    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


def _create_cross_encoder(model_name: str) -> Reranker:
    # Imported only when reranking is enabled and first used, keeping the
    # vector-only startup path lightweight.
    from sentence_transformers import CrossEncoder
    from torch import nn

    return CrossEncoder(model_name, activation_fn=nn.Sigmoid())


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        observability_service: ObservabilityService,
        reranker_factory: Callable[[str], Reranker] | None = None,
    ) -> None:
        self._top_k = settings.retrieval_top_k
        self._candidate_k = settings.retrieval_candidate_k
        self._score_threshold = settings.retrieval_score_threshold
        self._reranking_enabled = settings.reranking_enabled
        self._reranker_model = settings.reranker_model
        self._reranker_score_threshold = settings.reranker_score_threshold
        self._reranker_batch_size = settings.reranker_batch_size
        self._reranker_factory = reranker_factory or _create_cross_encoder
        self._reranker: Reranker | None = None
        self._observability = observability_service
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
                collection_metadata={"hnsw:space": "cosine"},
            )
            self._vectorstores[session_id] = vectorstore
        return vectorstore

    def _get_reranker(self) -> Reranker:
        with self._lock:
            if self._reranker is None:
                self._reranker = self._reranker_factory(self._reranker_model)
            return self._reranker

    @staticmethod
    def _with_score(
        document: Document,
        score_name: str,
        score: float,
    ) -> Document:
        return Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                score_name: round(score, 6),
                "relevance_score": round(score, 6),
            },
        )

    def _rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        if not candidates:
            return []

        pairs = [(query, document.page_content) for document in candidates]
        scores = self._get_reranker().predict(
            pairs,
            batch_size=self._reranker_batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results: list[Document] = []
        for document, raw_score in ranked:
            score = max(0.0, min(1.0, float(raw_score)))
            if score < self._reranker_score_threshold:
                continue
            results.append(self._with_score(document, "reranker_score", score))
            if len(results) == self._top_k:
                break
        return results

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

    def _retrieve(self, query: str, config: RunnableConfig) -> list[Document]:
        session_id = str(config.get("configurable", {}).get("session_id", "")).strip()
        if not session_id:
            return []

        with self._lock:
            vectorstore = self._vectorstores.get(session_id)
            if vectorstore is None:
                return []
            candidates = vectorstore.similarity_search_with_relevance_scores(
                query,
                k=self._candidate_k,
            )

        scored_candidates: list[Document] = []
        for document, score in candidates:
            normalized_score = max(0.0, min(1.0, float(score)))
            scored_candidates.append(
                self._with_score(document, "vector_score", normalized_score)
            )

        if self._reranking_enabled:
            return self._rerank(query, scored_candidates)

        results: list[Document] = []
        for document in scored_candidates:
            if document.metadata["vector_score"] < self._score_threshold:
                continue
            results.append(document)
            if len(results) == self._top_k:
                break
        return results

    def retrieve(self, query: str, config: RunnableConfig) -> list[Document]:
        started = perf_counter()
        results: list[Document] = []
        try:
            results = self._retrieve(query, config)
            return results
        except Exception as exc:
            self._observability.record_error(exc)
            raise
        finally:
            self._observability.record_retrieval(
                latency_ms=(perf_counter() - started) * 1000,
                document_count=len(results),
            )

    def list_documents(self, session_id: str) -> list[IndexedDocument]:
        with self._lock:
            return list(self._documents.get(session_id, []))
