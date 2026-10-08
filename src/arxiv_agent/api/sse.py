"""Server-sent events: one-way, over plain HTTP, reconnect built in.

Two shapes of stream:
- a Q&A answer: the work runs in a thread (the Answerer is synchronous) and
  reports each step through a callback; the callback hands it to the event
  loop, which sends it. The answer text comes last, with the result: a
  repair can still change it, so drafts are never streamed.
- a review job: it runs in the worker, which logs numbered events in
  Postgres. The stream replays the log after the id the browser last saw
  (EventSource sends it back as Last-Event-ID when it reconnects), then
  polls for new ones. Closing the tab never stops the review.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable

from sse_starlette import ServerSentEvent

from arxiv_agent.llm import LLMUnavailableError
from arxiv_agent.qa.answerer import NotIndexedError
from arxiv_agent.review.events import describe_event
from arxiv_agent.storage.job_store import JobStore

ENDED = {"done", "failed", "expired"}  # a paused job is waiting, not ended
POLL_SECONDS = 1.0

logger = logging.getLogger(__name__)

Emit = Callable[[str, dict], None]


def event(kind: str, data: dict, event_id: int | None = None) -> ServerSentEvent:
    return ServerSentEvent(
        data=json.dumps(data, default=str),
        event=kind,
        id=None if event_id is None else str(event_id),
    )


def _error(exc: BaseException) -> dict:
    # What the browser is told; the details go to the log.
    if isinstance(exc, NotIndexedError):
        return {"code": "not_indexed", "message": "Ingest this paper first."}
    if isinstance(exc, LLMUnavailableError):
        return {
            "code": "unavailable",
            "message": "The language models are unavailable; try again shortly.",
        }
    logger.error("stream failed", exc_info=exc)
    return {"code": "error", "message": "Something went wrong."}


async def stream_work(work: Callable[[Emit], dict]) -> AsyncIterator[ServerSentEvent]:
    # work(emit) runs in a thread; its steps stream as they happen, then
    # "result" (what it returned) or "error".
    loop = asyncio.get_running_loop()
    steps: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    def emit(kind: str, data: dict) -> None:
        loop.call_soon_threadsafe(steps.put_nowait, (kind, data))

    done = asyncio.ensure_future(asyncio.to_thread(work, emit))
    while True:
        step = asyncio.ensure_future(steps.get())
        await asyncio.wait({step, done}, return_when=asyncio.FIRST_COMPLETED)
        if step.done():
            yield event(*step.result())
            continue
        step.cancel()
        while not steps.empty():  # steps that arrived with the end
            yield event(*steps.get_nowait())
        if done.exception() is not None:
            yield event("error", _error(done.exception()))
        else:
            yield event("result", done.result())
        return


async def stream_job(
    jobs: JobStore, job_id: str, after: int = 0, poll: float = POLL_SECONDS
) -> AsyncIterator[ServerSentEvent]:
    # Every event after `after`, as it's logged; "end" once the job has ended
    # and nothing is left to send.
    while True:
        events = await asyncio.to_thread(jobs.events_after, job_id, after)
        for e in events:
            after = e.seq
            data = e.data | {"at": e.at, "text": describe_event(e.kind, e.data)}
            yield event(e.kind, data, event_id=e.seq)
        if not events:
            job = await asyncio.to_thread(jobs.get, job_id)
            if job is None or job.status in ENDED:
                yield event("end", {"status": job.status if job else "missing"})
                return
        await asyncio.sleep(poll)
