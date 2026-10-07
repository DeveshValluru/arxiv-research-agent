"""The review worker: takes queued review jobs and runs them, one at a time.

A review takes minutes, so it doesn't run inside the request that asks for it:
the request queues a job and returns its id, the worker runs it, and anyone
can follow its events, disconnect, and come back.

One worker for the whole system (scripts/review_worker.py holds a Postgres
advisory lock): reviews download from arXiv, which allows one connection at a
time. Jobs wait in the queue meanwhile; running more workers would first need
a rate limiter they all share.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from arxiv_agent.review.events import result_of
from arxiv_agent.review.graph import EventHandler
from arxiv_agent.review.state import Budget, ReviewState
from arxiv_agent.storage.job_store import Job, JobStore

# run(question, budget, on_event, job_id) -> (final state, trace id)
Runner = Callable[
    [str, Budget, EventHandler, str], Awaitable[tuple[ReviewState, str | None]]
]

logger = logging.getLogger(__name__)


class ReviewWorker:
    def __init__(
        self,
        jobs: JobStore,
        run: Runner,
        poll_seconds: float = 2.0,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        self._jobs = jobs
        self._run = run
        self._poll_seconds = poll_seconds
        self._heartbeat_seconds = heartbeat_seconds

    async def run_forever(self) -> None:
        while True:
            if not await self.run_next():
                await asyncio.sleep(self._poll_seconds)

    async def run_next(self) -> bool:
        job = self._jobs.claim_next()
        if job is None:
            return False
        await self.run_job(job)
        return True

    async def run_job(self, job: Job) -> None:
        # The heartbeat says "still alive" during long steps with no events.
        # If the worker process dies, the job stays 'running' until the next
        # worker starts and recover_orphans() puts it back in the queue.
        heartbeat = asyncio.create_task(self._beat(job.job_id))
        try:
            state, trace_id = await self._run(
                job.question,
                Budget.model_validate(job.budget),
                lambda kind, data: self._jobs.add_event(job.job_id, kind, data),
                job.job_id,
            )
        except Exception as exc:
            # One failed review mustn't stop the worker: record it, move on.
            logger.exception("review job %s failed", job.job_id)
            self._jobs.fail(job.job_id, f"{type(exc).__name__}: {exc}"[:1000])
        else:
            self._jobs.finish(job.job_id, result_of(state), trace_id)
        finally:
            heartbeat.cancel()

    async def _beat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            self._jobs.heartbeat(job_id)
