"""PDF parsing, metadata enrichment, and chunking."""

import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import Settings
from backend.models.responses import IndexedDocument
from backend.services.retrieval_service import RetrievalService


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def index_pdf(
        self,
        session_id: str,
        filename: str,
        content: bytes,
    ) -> IndexedDocument:
        safe_filename = Path(filename).name
        with tempfile.TemporaryDirectory(prefix="rag-upload-") as temp_dir:
            pdf_path = Path(temp_dir) / safe_filename
            pdf_path.write_bytes(content)
            documents = PyPDFLoader(str(pdf_path)).load()

        for document in documents:
            document.metadata["filename"] = safe_filename
            document.metadata["source"] = safe_filename

        chunks = self._text_splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No extractable text was found in the PDF")

        return self._retrieval_service.add_documents(
            session_id=session_id,
            filename=safe_filename,
            documents=chunks,
        )
