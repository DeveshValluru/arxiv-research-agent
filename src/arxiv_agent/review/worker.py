"""The review worker: takes queued review jobs and runs them, one at a time.

A review takes minutes, so it doesn't run inside the request that asks for it:
the request queues a job and returns its id, the worker runs it, and anyone
can follow its events, disconnect, and come back.

A job may run more than once: a deep review stops at the pause for a person
and is queued again with their decision; a job whose worker died is queued
again by the next worker. Each run continues from the job's last checkpoint
(the thread id is the job id).

One worker for the whole system (scripts/review_worker.py holds a Postgres
advisory lock): reviews download from arXiv, which allows one connection at a
time. Jobs wait in the queue meanwhile; running more workers would first need
a rate limiter they all share.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from arxiv_agent.review.events import result_of
from arxiv_agent.review.graph import EventHandler, ReviewRun
from arxiv_agent.storage.job_store import Job, JobStore

# run(job, on_event) -> the run's outcome: finished, or paused for review.
Runner = Callable[[Job, EventHandler], Awaitable[ReviewRun]]
# forget(job_id): delete a job's checkpoints once it can't continue any more.
Forget = Callable[[str], Awaitable[None]]

logger = logging.getLogger(__name__)


async def _keep_checkpoints(job_id: str) -> None:
    return None  # the default when there's no checkpointer to clean up


class ReviewWorker:
    def __init__(
        self,
        jobs: JobStore,
        run: Runner,
        forget: Forget = _keep_checkpoints,
        poll_seconds: float = 2.0,
        heartbeat_seconds: float = 15.0,
        review_hours: float = 24.0,
        sweep_seconds: float = 300.0,
    ) -> None:
        self._jobs = jobs
        self._run = run
        self._forget = forget
        self._poll_seconds = poll_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._review_hours = review_hours
        self._sweep_seconds = sweep_seconds

    async def run_forever(self) -> None:
        last_sweep = float("-inf")
        while True:
            if time.monotonic() - last_sweep > self._sweep_seconds:
                await self.sweep()
                last_sweep = time.monotonic()
            if not await self.run_next():
                await asyncio.sleep(self._poll_seconds)

    async def sweep(self) -> None:
        # Pauses nobody answered in time expire, and their checkpoints go.
        for job_id in self._jobs.expire_reviews(self._review_hours):
            await self._forget(job_id)

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

        def on_event(kind: str, data: dict) -> None:
            self._jobs.add_event(job.job_id, kind, data)
            # The reviewer's edits are eval labels: saved the moment they're
            # applied, so they're kept even if the review fails afterwards.
            if kind == "step" and data.get("edits"):
                self._jobs.save_labels(job.job_id, job.question, data["edits"])

        try:
            outcome = await self._run(job, on_event)
        except Exception as exc:
            # One failed review mustn't stop the worker: record it, move on.
            # Its checkpoints stay, so a retry continues where it failed.
            logger.exception("review job %s failed", job.job_id)
            self._jobs.fail(job.job_id, f"{type(exc).__name__}: {exc}"[:1000])
        else:
            if outcome.pause is not None:
                self._jobs.pause(job.job_id, outcome.pause)
            else:
                self._jobs.finish(
                    job.job_id, result_of(outcome.state), outcome.trace_id
                )
                await self._forget(job.job_id)  # the result is stored; done for good
        finally:
            heartbeat.cancel()

    async def _beat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            self._jobs.heartbeat(job_id)
