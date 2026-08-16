"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    hf_token: str | None
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-8b-instant"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    max_session_id_length: int = 128

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
            max_session_id_length=int(os.getenv("MAX_SESSION_ID_LENGTH", "128")),
        )

    def configure_environment(self) -> None:
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
