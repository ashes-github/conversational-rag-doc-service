"""Request-scoped metrics, request IDs, and structured application logs."""

from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda


@dataclass
class RequestMetrics:
    request_id: str
    method: str
    path: str
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    documents_retrieved: int = 0
    llm_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    status_code: int | None = None
    duration_ms: float | None = None
    error: str | None = None


class ObservabilityService:
    def __init__(self) -> None:
        self._logger = logging.getLogger("rag.observability")
        self._current: ContextVar[RequestMetrics | None] = ContextVar(
            "request_metrics",
            default=None,
        )

    @staticmethod
    def _request_id(request: Request) -> str:
        supplied = request.headers.get("X-Request-ID", "").strip()
        if supplied and len(supplied) <= 64 and re.fullmatch(r"[A-Za-z0-9._-]+", supplied):
            return supplied
        return str(uuid4())

    def begin(self, method: str, path: str, request_id: str) -> Token:
        return self._current.set(
            RequestMetrics(request_id=request_id, method=method, path=path)
        )

    def current_metrics(self) -> RequestMetrics | None:
        return self._current.get()

    def record_retrieval(self, latency_ms: float, document_count: int) -> None:
        metrics = self.current_metrics()
        if metrics is not None:
            metrics.retrieval_latency_ms += latency_ms
            metrics.documents_retrieved = document_count

    def record_error(self, error: Exception | str) -> None:
        metrics = self.current_metrics()
        if metrics is not None and metrics.error is None:
            text = str(error)
            metrics.error = text[:500] if text else type(error).__name__

    @staticmethod
    def _token_usage(message: Any) -> tuple[int | None, int | None, int | None]:
        usage = getattr(message, "usage_metadata", None) or {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        provider_usage = response_metadata.get("token_usage", {})

        input_tokens = usage.get("input_tokens", provider_usage.get("prompt_tokens"))
        output_tokens = usage.get(
            "output_tokens",
            provider_usage.get("completion_tokens"),
        )
        total_tokens = usage.get("total_tokens", provider_usage.get("total_tokens"))
        return input_tokens, output_tokens, total_tokens

    @staticmethod
    def _add_optional(current: int | None, additional: int | None) -> int | None:
        if additional is None:
            return current
        return (current or 0) + int(additional)

    def record_llm(self, latency_ms: float, message: Any) -> None:
        metrics = self.current_metrics()
        if metrics is None:
            return
        input_tokens, output_tokens, total_tokens = self._token_usage(message)
        metrics.llm_latency_ms += latency_ms
        metrics.llm_calls += 1
        metrics.input_tokens = self._add_optional(metrics.input_tokens, input_tokens)
        metrics.output_tokens = self._add_optional(metrics.output_tokens, output_tokens)
        metrics.total_tokens = self._add_optional(metrics.total_tokens, total_tokens)

    def wrap_llm(self, model: Any) -> Runnable:
        def invoke(input_value: Any, config: RunnableConfig) -> Any:
            started = perf_counter()
            try:
                message = model.invoke(input_value, config=config)
            except Exception as exc:
                self.record_error(exc)
                raise
            self.record_llm((perf_counter() - started) * 1000, message)
            return message

        async def ainvoke(input_value: Any, config: RunnableConfig) -> Any:
            started = perf_counter()
            try:
                message = await model.ainvoke(input_value, config=config)
            except Exception as exc:
                self.record_error(exc)
                raise
            self.record_llm((perf_counter() - started) * 1000, message)
            return message

        return RunnableLambda(invoke, afunc=ainvoke)

    async def middleware(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = self._request_id(request)
        token = self.begin(request.method, request.url.path, request_id)
        started = perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            self.record_error(exc)
            self._logger.exception(
                "Unhandled request error request_id=%s path=%s",
                request_id,
                request.url.path,
            )
            raise
        finally:
            metrics = self.current_metrics()
            if metrics is not None:
                metrics.duration_ms = round((perf_counter() - started) * 1000, 3)
                metrics.status_code = response.status_code if response is not None else 500
                if metrics.status_code >= 400 and metrics.error is None:
                    metrics.error = f"HTTP {metrics.status_code}"
                payload = {
                    key: value
                    for key, value in asdict(metrics).items()
                    if value is not None
                }
                self._logger.info(json.dumps({"event": "request_completed", **payload}))
            self._current.reset(token)
