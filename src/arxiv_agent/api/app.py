"""The routes. Everything slow runs in a thread; everything long streams.

    python -m uvicorn arxiv_agent.api.app:app --port 8000

- GET  /api/health                      the API, and whether the worker runs
- GET  /api/papers/search?q=            arXiv search (or one id), as cards
- GET  /api/papers/{id}                 one paper, stored or from arXiv
- POST /api/papers/{id}/ingest          fetch, chunk, embed (seconds)
- POST /api/ask                         a question about one paper, as SSE
- POST /api/reviews                     queue a literature review
- GET  /api/reviews                     recent reviews
- GET  /api/reviews/{id}                one review: status, result, pause
- GET  /api/reviews/{id}/events         its progress, as SSE (resumable)
- POST /api/reviews/{id}/decision       continue a paused review
- POST /api/reviews/{id}/retry          queue a failed review again
- GET  /api/reviews/{id}/graph          its citation graph
- GET/PUT/DELETE /api/reading-list...   saved papers

Local demo: no auth, one user. Input limits still apply (question
lengths), and every Q&A trace is tagged "api".
"""

import asyncio
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from langfuse import propagate_attributes
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette import EventSourceResponse

from arxiv_agent.api.graph import Graph, citation_graph
from arxiv_agent.api.services import Services, build_services
from arxiv_agent.api.sse import Emit, stream_job, stream_work
from arxiv_agent.clients.arxiv import (
    ARXIV_ID_PATTERN,
    ArxivError,
    ArxivNotFoundError,
    PaperSummary,
)
from arxiv_agent.mcp_servers.arxiv_server import build_search_query
from arxiv_agent.review.state import Budget
from arxiv_agent.storage.job_store import Job
from arxiv_agent.storage.reading_list import SavedPaper

WEB_ORIGIN = os.environ.get("WEB_ORIGIN", "http://localhost:3000")
PING_SECONDS = 15  # keeps idle streams (a paused review) open through proxies
VERSION = re.compile(r"v\d+$")


class PaperView(BaseModel):
    # A paper as the UI shows it, with what we hold of it.
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    categories: list[str]
    indexed: bool  # its chunks are in the index: it can be asked about
    saved: bool  # on the reading list


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)
    arxiv_id: str = Field(pattern=ARXIV_ID_PATTERN.pattern)
    version: int = Field(ge=1)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=10, max_length=300)
    # deep: pause after screening for the person to check the paper list
    deep: bool = False


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    remove: list[str] = []
    add: list[str] = []


class ReviewView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: Job
    result: dict | None  # once done
    review_request: dict | None  # while it waits for the person


def services(request: Request) -> Services:
    return request.app.state.services


Svc = Depends(services)
router = APIRouter(prefix="/api")


def _view(s: Services, paper: PaperSummary, saved: set[str]) -> PaperView:
    indexed = s.chunks.stored_chunker_version(paper.arxiv_id, paper.version)
    return PaperView(
        **paper.model_dump(exclude={"updated", "primary_category"}),
        indexed=indexed is not None,
        saved=paper.arxiv_id in saved,
    )


def _arxiv_failed(exc: ArxivError) -> HTTPException:
    return HTTPException(503, f"arXiv: {exc}")


@router.get("/health")
async def health(s: Services = Svc) -> dict:
    return {"ok": True, "worker": await asyncio.to_thread(s.jobs.worker_running)}


@router.get("/papers/search")
async def search_papers(
    q: str, limit: int = Query(10, ge=1, le=25), s: Services = Svc
) -> list[PaperView]:
    def search() -> list[PaperView]:
        query = q.strip()
        if ARXIV_ID_PATTERN.match(query):  # an id: that paper
            found = s.arxiv.get_metadata([VERSION.sub("", query)])
            papers = list(found.values())
        else:
            papers = s.arxiv.search_papers(build_search_query(query), limit)
        saved = s.reading.ids()
        return [_view(s, paper, saved) for paper in papers]

    try:
        return await asyncio.to_thread(search)
    except ValueError as exc:  # no keywords left once stopwords are dropped
        raise HTTPException(422, str(exc)) from exc
    except ArxivError as exc:
        raise _arxiv_failed(exc) from exc


def _find(s: Services, arxiv_id: str, version: int | None) -> PaperSummary:
    stored = s.chunks.get_paper(arxiv_id, version) if version else None
    if stored:
        return stored
    paper = s.arxiv.get_metadata([arxiv_id]).get(arxiv_id)
    if paper is None:
        raise HTTPException(404, f"arXiv has no paper {arxiv_id}")
    # arXiv describes the latest version; it stands in for an older one.
    return paper.model_copy(update={"version": version}) if version else paper


@router.get("/papers/{arxiv_id:path}")
async def get_paper(
    arxiv_id: str, version: int | None = None, s: Services = Svc
) -> PaperView:
    def find() -> PaperView:
        return _view(s, _find(s, arxiv_id, version), s.reading.ids())

    try:
        return await asyncio.to_thread(find)
    except ArxivError as exc:
        raise _arxiv_failed(exc) from exc


class Ingested(BaseModel):
    source: str  # full_text, abstract_only or unavailable
    paper: PaperView


@router.post("/papers/{arxiv_id:path}/ingest")
async def ingest_paper(
    arxiv_id: str, version: int | None = None, s: Services = Svc
) -> Ingested:
    def ingest() -> Ingested:
        paper = _find(s, arxiv_id, version)
        source = s.library.ensure_ingested(paper.arxiv_id, paper.version)
        return Ingested(source=source, paper=_view(s, paper, s.reading.ids()))

    try:
        return await asyncio.to_thread(ingest)
    except ArxivNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ArxivError as exc:
        raise _arxiv_failed(exc) from exc


@router.post("/ask")
async def ask(body: AskRequest, s: Services = Svc) -> EventSourceResponse:
    def work(emit: Emit) -> dict:
        with propagate_attributes(user_id="local", tags=["api"]):
            result = s.answerer.ask(
                body.question, body.arxiv_id, body.version, on_event=emit
            )
        return result.model_dump(mode="json")

    return EventSourceResponse(stream_work(work), ping=PING_SECONDS)


@router.post("/reviews", status_code=201)
async def submit_review(body: ReviewRequest, s: Services = Svc) -> dict:
    job_id = await asyncio.to_thread(
        s.jobs.submit, body.question, Budget().model_dump(), body.deep
    )
    return {"job_id": job_id}


@router.get("/reviews")
async def list_reviews(
    limit: int = Query(20, ge=1, le=100), s: Services = Svc
) -> list[Job]:
    return await asyncio.to_thread(s.jobs.recent, limit)


@router.get("/reviews/{job_id}")
async def get_review(job_id: uuid.UUID, s: Services = Svc) -> ReviewView:
    def read() -> ReviewView:
        job = s.jobs.get(str(job_id))
        if job is None:
            raise HTTPException(404, "no such review")
        waiting = job.status == "awaiting_review"
        return ReviewView(
            job=job,
            result=s.jobs.result(str(job_id)) if job.status == "done" else None,
            review_request=s.jobs.review_request(str(job_id)) if waiting else None,
        )

    return await asyncio.to_thread(read)


@router.get("/reviews/{job_id}/events")
async def review_events(
    job_id: uuid.UUID,
    after: int = 0,
    last_event_id: str | None = Header(None),
    s: Services = Svc,
) -> EventSourceResponse:
    # EventSource sends Last-Event-ID when it reconnects: resume from there.
    if last_event_id and last_event_id.isdigit():
        after = int(last_event_id)
    if await asyncio.to_thread(s.jobs.get, str(job_id)) is None:
        raise HTTPException(404, "no such review")
    return EventSourceResponse(
        stream_job(s.jobs, str(job_id), after), ping=PING_SECONDS
    )


@router.post("/reviews/{job_id}/decision")
async def decide(job_id: uuid.UUID, body: Decision, s: Services = Svc) -> dict:
    if not await asyncio.to_thread(s.jobs.decide, str(job_id), body.model_dump()):
        raise HTTPException(409, "this review isn't waiting for a decision")
    return {"status": "queued"}


@router.post("/reviews/{job_id}/retry")
async def retry(job_id: uuid.UUID, s: Services = Svc) -> dict:
    if not await asyncio.to_thread(s.jobs.retry, str(job_id)):
        raise HTTPException(409, "only a failed review can be retried")
    return {"status": "queued"}


@router.get("/reviews/{job_id}/graph")
async def review_graph(job_id: uuid.UUID, s: Services = Svc) -> Graph:
    result = await asyncio.to_thread(s.jobs.result, str(job_id))
    if not result:
        raise HTTPException(404, "no finished review with that id")
    return citation_graph(result)


@router.get("/reading-list")
async def reading_list(s: Services = Svc) -> list[SavedPaper]:
    return await asyncio.to_thread(s.reading.all)


@router.put("/reading-list/{arxiv_id:path}")
async def save_paper(arxiv_id: str, body: SavedPaper, s: Services = Svc) -> SavedPaper:
    if body.arxiv_id != arxiv_id:
        raise HTTPException(422, "the id in the path and the body differ")
    return await asyncio.to_thread(s.reading.save, body)


@router.delete("/reading-list/{arxiv_id:path}", status_code=204)
async def unsave_paper(arxiv_id: str, s: Services = Svc) -> None:
    if not await asyncio.to_thread(s.reading.remove, arxiv_id):
        raise HTTPException(404, "not on the reading list")


def create_app(built: Services | None = None) -> FastAPI:
    # built: services made elsewhere (tests); otherwise made at startup.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = built or await asyncio.to_thread(build_services)
        yield
        if built is None:
            app.state.services.close()

    app = FastAPI(title="arXiv Research Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[WEB_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
