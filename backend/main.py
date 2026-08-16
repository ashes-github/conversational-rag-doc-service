"""FastAPI application assembly."""

from fastapi import FastAPI
from langserve import add_routes

from backend.api.chat import create_chat_router
from backend.api.documents import create_documents_router
from backend.core.config import Settings
from backend.core.logging import configure_logging
from backend.services.conversation_service import ConversationService
from backend.services.ingestion_service import IngestionService
from backend.services.rag_service import RagService
from backend.services.retrieval_service import RetrievalService


def create_app() -> FastAPI:
    configure_logging()
    settings = Settings.from_environment()
    settings.configure_environment()

    conversation_service = ConversationService()
    retrieval_service = RetrievalService(settings)
    ingestion_service = IngestionService(settings, retrieval_service)
    rag_service = RagService(settings, retrieval_service, conversation_service)

    app = FastAPI(
        title="Conversational RAG API",
        version="1.2.0",
        description="Conversational question answering over session-scoped PDF uploads",
    )
    app.include_router(
        create_documents_router(settings, ingestion_service, retrieval_service)
    )
    app.include_router(create_chat_router(conversation_service))
    add_routes(app, rag_service.chain, path="/chain")

    app.state.settings = settings
    app.state.conversation_service = conversation_service
    app.state.retrieval_service = retrieval_service
    app.state.ingestion_service = ingestion_service
    app.state.rag_service = rag_service
    return app


app = create_app()
