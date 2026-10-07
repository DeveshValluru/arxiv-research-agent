"""First Langfuse trace: one LLM call wrapped in a span.

    uv run --env-file .env python scripts/trace_hello.py

get_client() reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL.
"""

from huggingface_hub import InferenceClient
from langfuse import get_client

langfuse = get_client()
client = InferenceClient(provider="deepinfra")
MODEL = "Qwen/Qwen3-32B"
QUESTION = "What is position bias when an LLM is used as a judge?"
messages = [
    {"role": "system", "content": "Answer in one sentence. /no_think"},
    {"role": "user", "content": QUESTION},
]

with langfuse.start_as_current_observation(
    as_type="span", name="hello", input=QUESTION
) as span:
    with langfuse.start_as_current_observation(
        as_type="generation", name="llm", model=MODEL, input=messages
    ) as generation:
        response = client.chat_completion(messages, model=MODEL, max_tokens=500)
        answer = response.choices[0].message.content.strip()
        cost = response.usage.estimated_cost
        generation.update(
            output=answer,
            usage_details={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
            cost_details={"total": cost} if cost is not None else None,
        )
    span.update(output=answer)

# Traces are sent in background batches; a short script must flush before it
# exits, or the batch is lost.
langfuse.flush()
