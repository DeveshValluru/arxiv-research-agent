"""First hosted LLM call through Hugging Face Inference Providers.

    uv run --env-file .env python scripts/llm_hello.py

InferenceClient reads HF_TOKEN from the environment, which is why --env-file is needed.
"""

from huggingface_hub import InferenceClient

client = InferenceClient(provider="deepinfra")  # WHO runs the model

response = client.chat_completion(
    model="Qwen/Qwen3-32B",
    messages=[
        {
            "role": "system",
            "content": "You are a concise research assistant. Answer in one sentence. /no_think",
        },
        {
            "role": "user",
            "content": "What is position bias when an LLM is used as a judge?",
        },
    ],
    max_tokens=500,
)

print(response.choices[0].message.content)
print(response.usage)
print(response.choices[0].finish_reason)
