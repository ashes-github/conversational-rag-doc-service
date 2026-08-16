"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _environment_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    hf_token: str | None
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-8b-instant"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    retrieval_candidate_k: int = 8
    retrieval_score_threshold: float = 0.20
    reranking_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_score_threshold: float = 0.10
    reranker_batch_size: int = 8
    max_session_id_length: int = 128

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be between zero and CHUNK_SIZE")
        if self.retrieval_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be greater than zero")
        if self.retrieval_candidate_k < self.retrieval_top_k:
            raise ValueError("RETRIEVAL_CANDIDATE_K must be at least RETRIEVAL_TOP_K")
        if not 0.0 <= self.retrieval_score_threshold <= 1.0:
            raise ValueError("RETRIEVAL_SCORE_THRESHOLD must be between 0 and 1")
        if not 0.0 <= self.reranker_score_threshold <= 1.0:
            raise ValueError("RERANKER_SCORE_THRESHOLD must be between 0 and 1")
        if self.reranker_batch_size <= 0:
            raise ValueError("RERANKER_BATCH_SIZE must be greater than zero")

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            hf_token=os.getenv("HF_TOKEN"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
            retrieval_candidate_k=int(os.getenv("RETRIEVAL_CANDIDATE_K", "8")),
            retrieval_score_threshold=float(
                os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.20")
            ),
            reranking_enabled=_environment_bool("RERANKING_ENABLED", False),
            reranker_model=os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L6-v2",
            ),
            reranker_score_threshold=float(
                os.getenv("RERANKER_SCORE_THRESHOLD", "0.10")
            ),
            reranker_batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
            max_session_id_length=int(os.getenv("MAX_SESSION_ID_LENGTH", "128")),
        )

    def configure_environment(self) -> None:
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
