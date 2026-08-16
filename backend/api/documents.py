"""Document upload and listing routes."""

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.core.config import Settings
from backend.models.responses import DocumentsResponse, IndexedDocument, UploadResponse
from backend.services.ingestion_service import IngestionService
from backend.services.retrieval_service import RetrievalService


def _validate_session_id(session_id: str, settings: Settings) -> str:
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id must not be empty")
    if len(session_id) > settings.max_session_id_length:
        raise HTTPException(status_code=422, detail="session_id is too long")
    return session_id


def create_documents_router(
    settings: Settings,
    ingestion_service: IngestionService,
    retrieval_service: RetrievalService,
) -> APIRouter:
    router = APIRouter(tags=["documents"])

    @router.post("/upload", response_model=UploadResponse)
    async def upload_documents(
        files: List[UploadFile] = File(...),
        session_id: str = Form(...),
    ) -> UploadResponse:
        validated_session_id = _validate_session_id(session_id, settings)
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF is required")

        indexed_files: list[IndexedDocument] = []
        for uploaded_file in files:
            filename = Path(uploaded_file.filename or "").name
            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are supported: {filename or 'unnamed file'}",
                )

            content = await uploaded_file.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"{filename} is empty")

            try:
                indexed = await run_in_threadpool(
                    ingestion_service.index_pdf,
                    validated_session_id,
                    filename,
                    content,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not process {filename}: {exc}",
                ) from exc
            indexed_files.append(indexed)

        return UploadResponse(
            session_id=validated_session_id,
            files_uploaded=len(indexed_files),
            chunks_indexed=sum(document.chunks for document in indexed_files),
            documents=indexed_files,
        )

    @router.get("/documents/{session_id}", response_model=DocumentsResponse)
    async def get_documents(session_id: str) -> DocumentsResponse:
        return DocumentsResponse(
            session_id=session_id,
            documents=retrieval_service.list_documents(session_id),
        )

    return router
