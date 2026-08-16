"""Session-scoped in-memory conversation history."""

from threading import RLock

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


class ConversationService:
    def __init__(self) -> None:
        self._histories: dict[str, ChatMessageHistory] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> BaseChatMessageHistory:
        with self._lock:
            if session_id not in self._histories:
                self._histories[session_id] = ChatMessageHistory()
            return self._histories[session_id]

    def get_messages(self, session_id: str) -> list[dict[str, str]] | None:
        with self._lock:
            history = self._histories.get(session_id)
            if history is None:
                return None
            return [
                {"type": message.type, "content": str(message.content)}
                for message in history.messages
            ]
