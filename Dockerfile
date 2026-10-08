# syntax=docker/dockerfile:1
# The API and the review worker: one image, two commands (compose.yaml).
# Dependencies come from uv.lock exactly; on Linux that means CPU-only torch
# from PyTorch's index (pyproject.toml), no CUDA libraries.
FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir uv==0.11.26 \
    && useradd --create-home app \
    && mkdir -p /app/data/html /cache/huggingface \
    && chown -R app /app /cache

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

# Dependencies first: this layer is reused until uv.lock changes. uv's
# download cache lives in a build cache mount, not in the image (~1.3 GB).
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

USER app
EXPOSE 8000
CMD ["uvicorn", "arxiv_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
