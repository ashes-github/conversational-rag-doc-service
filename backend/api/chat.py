"""Conversation history routes."""

from fastapi import APIRouter, HTTPException

from backend.models.responses import HistoryResponse
from backend.services.conversation_service import ConversationService


def create_chat_router(conversation_service: ConversationService) -> APIRouter:
    router = APIRouter(tags=["chat"])

    @router.get("/history/{session_id}", response_model=HistoryResponse)
    async def get_history(session_id: str) -> HistoryResponse:
        messages = conversation_service.get_messages(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return HistoryResponse(session_id=session_id, messages=messages)

    return router
