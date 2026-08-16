"""Conversational RAG chain construction and orchestration."""

from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from backend.core.config import Settings
from backend.services.conversation_service import ConversationService
from backend.services.guardrail_service import GuardrailService
from backend.services.observability_service import ObservabilityService
from backend.services.retrieval_service import RetrievalService


class RagService:
    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        conversation_service: ConversationService,
        guardrail_service: GuardrailService,
        observability_service: ObservabilityService,
    ) -> None:
        llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
        )
        measured_llm = observability_service.wrap_llm(llm)
        session_retriever = RunnableLambda(retrieval_service.retrieve)

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given a chat history and the latest user question, formulate a "
                    "standalone question that can be understood without the chat "
                    "history. Do not answer the question. Return it unchanged if "
                    "reformulation is not needed.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            measured_llm,
            session_retriever,
            contextualize_prompt,
        )

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer questions only from the retrieved document context "
                    "below. If the context does not contain the answer, say that you "
                    "couldn't find the answer in the uploaded documents. Use at most "
                    "three concise sentences.\n\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        answer_chain = create_stuff_documents_chain(measured_llm, answer_prompt)
        rag_chain = guardrail_service.build_chain(
            history_aware_retriever,
            answer_chain,
        )

        self.chain = RunnableWithMessageHistory(
            rag_chain,
            conversation_service.get_or_create,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
