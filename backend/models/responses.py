"""Typed API response payloads."""

from pydantic import BaseModel


class IndexedDocument(BaseModel):
    filename: str
    chunks: int


class UploadResponse(BaseModel):
    session_id: str
    files_uploaded: int
    chunks_indexed: int
    documents: list[IndexedDocument]


class DocumentsResponse(BaseModel):
    session_id: str
    documents: list[IndexedDocument]


class HistoryMessage(BaseModel):
    type: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
