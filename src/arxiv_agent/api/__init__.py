"""The HTTP API (Phase 8): what the web UI talks to.

    python -m uvicorn arxiv_agent.api.app:app --port 8000

Paper Q&A streamed over SSE, review jobs (submit, watch, decide), paper search
and ingestion, a reading list, and a review's citation graph. Reviews run in
scripts/review_worker.py; the API only queues and watches them.
"""
