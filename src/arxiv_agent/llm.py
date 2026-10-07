import asyncio
import random
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError
from langfuse import Langfuse, get_client
from pydantic import BaseModel, ValidationError

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
ATTEMPTS_PER_PROVIDER = 2
BACKOFF_SECONDS = 2.0
THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LLMUnavailableError(Exception):
    pass


class LLMOutputError(Exception):
    pass


def is_transient(exc: Exception) -> bool:
    # Worth retrying: network trouble, timeouts, rate limits, overloaded or broken
    # servers. Not worth it: our own mistakes (a bad request, a wrong model name).
    if isinstance(exc, (httpx.TransportError, InferenceTimeoutError)):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in RETRYABLE_STATUSES


def chat_with_failover(
    *,
    clients: dict[str, InferenceClient],
    providers: list[str],
    model: str,
    messages: list[dict],
    langfuse: Langfuse,
    name: str = "llm",
    metadata: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
    **params: Any,
) -> tuple[Any, str, int]:
    # Retry transient failures with exponential backoff and jitter, then fail
    # over to the next provider serving the same model. Each attempt is its own
    # generation in the trace, so failovers are visible, not mysterious.
    # Returns the response, the provider that answered, and the attempt count.
    attempts = 0
    failures: list[str] = []
    for provider in providers:
        for attempt in range(1, ATTEMPTS_PER_PROVIDER + 1):
            attempts += 1
            with langfuse.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=messages,
                model_parameters=params,
                metadata={
                    **(metadata or {}),
                    "provider": provider,
                    "attempt": attempts,
                },
            ) as generation:
                try:
                    response = clients[provider].chat_completion(
                        messages, model=model, **params
                    )
                except Exception as exc:
                    if not is_transient(exc):
                        raise
                    generation.update(
                        level="ERROR", status_message=f"{provider}: {exc}"[:500]
                    )
                    failures.append(f"{provider}: {type(exc).__name__}")
                else:
                    cost = getattr(response.usage, "estimated_cost", None)
                    generation.update(
                        output=response.choices[0].message.content,
                        usage_details={
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens,
                        },
                        cost_details={"total": cost} if cost is not None else None,
                    )
                    return response, provider, attempts
            if attempt < ATTEMPTS_PER_PROVIDER:
                backoff = BACKOFF_SECONDS * 2 ** (attempt - 1)
                sleep(backoff + random.uniform(0, 1))

    raise LLMUnavailableError(f"{model}: every provider failed ({'; '.join(failures)})")


def strip_thinking(text: str) -> str:
    return THINKING.sub("", text).strip()


def parse_json_object[T: BaseModel](raw: str, schema: type[T]) -> T:
    # Lenient about wrapping (code fences, a sentence around it), strict about
    # the content: it must validate against the schema.
    match = JSON_OBJECT.search(strip_thinking(raw))
    if match is None:
        raise LLMOutputError(f"no JSON object in the model's reply: {raw[:200]!r}")
    try:
        return schema.model_validate_json(match.group())
    except ValidationError as exc:
        problems = "; ".join(e["msg"] for e in exc.errors()[:3])
        raise LLMOutputError(
            f"the model's reply doesn't fit {schema.__name__}: {problems}"
        ) from exc


class ChatModel:
    # One model on one or more providers, callable from async code (graph
    # nodes). The blocking HTTP call runs in a worker thread so the event loop
    # stays free; the thread inherits the trace context, so spans still nest.
    def __init__(
        self,
        model: str,
        providers: list[str],
        clients: dict[str, InferenceClient] | None = None,
        langfuse: Langfuse | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._providers = providers
        self._clients = clients or {p: InferenceClient(provider=p) for p in providers}
        self._langfuse = langfuse or get_client()
        self._sleep = sleep

    async def complete(
        self,
        messages: list[dict],
        *,
        name: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str:
        response, _, _ = await asyncio.to_thread(
            chat_with_failover,
            clients=self._clients,
            providers=self._providers,
            model=self.model,
            messages=messages,
            langfuse=self._langfuse,
            name=name,
            sleep=self._sleep,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return strip_thinking(response.choices[0].message.content or "")
