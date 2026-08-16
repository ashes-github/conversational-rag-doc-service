"""Deterministic grounding guardrails for the RAG answer path."""

from typing import Any

from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)


class GuardrailService:
    no_context_response = "I couldn't find this in the uploaded documents."

    @staticmethod
    def has_relevant_context(payload: dict[str, Any]) -> bool:
        return bool(payload.get("context"))

    def fallback_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "answer": self.no_context_response,
            "grounded": False,
        }

    @staticmethod
    def mark_grounded(payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "grounded": True}

    def build_chain(
        self,
        retriever: Runnable,
        answer_chain: Runnable,
    ) -> Runnable:
        retrieve_context = RunnablePassthrough.assign(context=retriever)
        generate_grounded_answer = (
            RunnablePassthrough.assign(answer=answer_chain)
            | RunnableLambda(self.mark_grounded)
        )
        return retrieve_context | RunnableBranch(
            (
                self.has_relevant_context,
                generate_grounded_answer,
            ),
            RunnableLambda(self.fallback_response),
        )
