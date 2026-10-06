import random
import time
from collections.abc import Callable
from typing import Any

import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError
from langfuse import Langfuse

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
ATTEMPTS_PER_PROVIDER = 2
BACKOFF_SECONDS = 2.0


class LLMUnavailableError(Exception):
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
