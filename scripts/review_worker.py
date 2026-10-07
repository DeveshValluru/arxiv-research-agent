"""Run review jobs in the background: the worker.

    uv run --env-file .env python scripts/review_worker.py

Takes queued jobs (scripts/review_jobs.py submit) one at a time, until you
stop it with Ctrl+C. Only one worker can run: reviews download from arXiv,
which allows one connection at a time. A job interrupted by stopping the
worker goes back in the queue when the worker starts again, and continues
from its last finished step.
"""

import asyncio
import logging
import os
import sys

from langfuse import get_client

from arxiv_agent.review.checkpoints import postgres_checkpointer
from arxiv_agent.review.graph import EventHandler, ReviewRun, run_review
from arxiv_agent.review.setup import ReviewSettings, open_review_graph
from arxiv_agent.review.state import Budget
from arxiv_agent.review.worker import ReviewWorker
from arxiv_agent.storage.job_store import Job, JobStore


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

    # Checkpoints in Postgres: a paused job, or one whose worker died, carries
    # on from where it stopped (the thread id is the job id).
    with postgres_checkpointer(os.environ["DATABASE_URL"]) as checkpointer:
        async with open_review_graph(ReviewSettings(), langfuse, checkpointer) as graph:

            async def run(job: Job, on_event: EventHandler) -> ReviewRun:
                try:
                    return await run_review(
                        job.question,
                        graph,
                        thread_id=job.job_id,
                        budget=Budget.model_validate(job.budget),
                        pause_for_review=job.pause_for_review,
                        decision=job.decision,
                        on_event=on_event,
                        langfuse=langfuse,
                    )
                finally:
                    langfuse.flush()

            worker = ReviewWorker(jobs, run, forget=checkpointer.adelete_thread)
            print("Worker ready, waiting for jobs. Ctrl+C to stop.", flush=True)
            await worker.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Worker stopped. A job it was running continues when it restarts.")
