"""FastAPI application assembly."""

from fastapi import FastAPI
from langserve import add_routes

from backend.api.chat import create_chat_router
from backend.api.documents import create_documents_router
from backend.core.config import Settings
from backend.core.logging import configure_logging
from backend.services.conversation_service import ConversationService
from backend.services.guardrail_service import GuardrailService
from backend.services.ingestion_service import IngestionService
from backend.services.observability_service import ObservabilityService
from backend.services.rag_service import RagService
from backend.services.retrieval_service import RetrievalService


def create_app() -> FastAPI:
    configure_logging()
    settings = Settings.from_environment()
    settings.configure_environment()

    observability_service = ObservabilityService()
    conversation_service = ConversationService()
    retrieval_service = RetrievalService(settings, observability_service)
    ingestion_service = IngestionService(settings, retrieval_service)
    guardrail_service = GuardrailService()
    rag_service = RagService(
        settings,
        retrieval_service,
        conversation_service,
        guardrail_service,
        observability_service,
    )

    app = FastAPI(
        title="Conversational RAG API",
        version="1.2.0",
        description="Conversational question answering over session-scoped PDF uploads",
    )
    app.middleware("http")(observability_service.middleware)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Return a lightweight readiness response for container health checks."""
        return {"status": "ok"}

    app.include_router(
        create_documents_router(settings, ingestion_service, retrieval_service)
    )
    app.include_router(create_chat_router(conversation_service))
    add_routes(app, rag_service.chain, path="/chain")

    app.state.settings = settings
    app.state.observability_service = observability_service
    app.state.conversation_service = conversation_service
    app.state.retrieval_service = retrieval_service
    app.state.ingestion_service = ingestion_service
    app.state.guardrail_service = guardrail_service
    app.state.rag_service = rag_service
    return app


app = create_app()
