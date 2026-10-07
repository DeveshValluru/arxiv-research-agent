"""Run review jobs in the background: the worker.

    uv run --env-file .env python scripts/review_worker.py

Takes queued jobs (scripts/review_jobs.py submit) one at a time, until you
stop it with Ctrl+C. Only one worker can run: reviews download from arXiv,
which allows one connection at a time. A job interrupted by stopping the
worker goes back in the queue when the worker starts again.
"""

import asyncio
import logging
import os
import sys

from langfuse import get_client

from arxiv_agent.review.graph import EventHandler, run_review
from arxiv_agent.review.setup import ReviewSettings, open_review_graph
from arxiv_agent.review.state import Budget
from arxiv_agent.review.worker import ReviewWorker
from arxiv_agent.storage.job_store import JobStore


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    jobs = JobStore.connect(os.environ["DATABASE_URL"])
    # Checked before the slow setup, so a second worker fails fast.
    if not jobs.try_lock_worker():
        sys.exit(
            "Another review worker is already running (only one at a time: "
            "arXiv allows one connection)."
        )
    for job_id, status in jobs.recover_orphans().items():
        print(
            f"job {job_id} was interrupted by a stopped worker: now {status}",
            flush=True,
        )

    langfuse = get_client()

    async with open_review_graph(ReviewSettings(), langfuse) as graph:

        async def run(
            question: str, budget: Budget, on_event: EventHandler, job_id: str
        ):
            try:
                return await run_review(
                    question,
                    graph,
                    budget=budget,
                    on_event=on_event,
                    session_id=job_id,
                    langfuse=langfuse,
                )
            finally:
                langfuse.flush()

        print("Worker ready, waiting for jobs. Ctrl+C to stop.", flush=True)
        await ReviewWorker(jobs, run).run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Worker stopped. A job it was running will be retried on restart.")
