import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from arxiv_agent.api.app import create_app
from arxiv_agent.api.graph import citation_graph
from arxiv_agent.api.services import Services
from arxiv_agent.clients.arxiv import ArxivUnavailableError, PaperSummary
from arxiv_agent.qa.answerer import NotIndexedError
from arxiv_agent.storage.reading_list import ReadingList
from tests.test_runner import make_result

PAPER = PaperSummary(
    arxiv_id="2411.15594",
    version=6,
    title="A Survey on LLM-as-a-Judge",
    authors=["Jiawei Gu"],
    abstract="Judging with LLMs.",
    published=datetime(2024, 11, 23, tzinfo=UTC),
    updated=datetime(2025, 3, 1, tzinfo=UTC),
    primary_category="cs.CL",
    categories=["cs.CL"],
)


class FakeArxiv:
    def __init__(self, down: bool = False) -> None:
        self.down = down
        self.queries: list[str] = []

    def search_papers(self, query, max_results=5):
        if self.down:
            raise ArxivUnavailableError("arXiv unavailable after 4 attempts: HTTP 503")
        self.queries.append(query)
        return [PAPER]

    def get_metadata(self, ids):
        return {PAPER.arxiv_id: PAPER} if PAPER.arxiv_id in ids else {}


class FakeLibrary:
    def __init__(self) -> None:
        self.ingested: list[tuple[str, int]] = []

    def ensure_ingested(self, arxiv_id, version):
        self.ingested.append((arxiv_id, version))
        return "full_text"


class FakeAnswerer:
    # Reports the steps the real one does; papers it doesn't know aren't
    # indexed.
    def ask(self, question, arxiv_id, version, on_event=None):
        if arxiv_id != PAPER.arxiv_id:
            raise NotIndexedError("ingest it first")
        on_event("retrieved", {"sources": [{"label": "S1", "section": "Intro"}]})
        on_event("answered", {"status": "answered"})
        on_event("checked", {"cited": 1, "failed": 0, "repair": None})
        return make_result("Swapping reduces bias [S1].", "answered")


@pytest.fixture
def client(store, jobs, connect):
    services = Services(
        chunks=store,
        jobs=jobs,
        reading=ReadingList(connect()),
        arxiv=FakeArxiv(),
        library=FakeLibrary(),
        answerer=FakeAnswerer(),
    )
    with TestClient(create_app(services)) as client:
        client.services = services
        yield client


def events(response) -> list[tuple[str, dict, str | None]]:
    # Server-sent events as (event, data, id); pings are skipped.
    found = []
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "event" in fields:
            found.append(
                (fields["event"], json.loads(fields["data"]), fields.get("id"))
            )
    return found


def test_health_says_whether_a_review_worker_is_running(client):
    body = client.get("/api/health").json()

    assert body["ok"] is True and isinstance(body["worker"], bool)


def test_search_returns_cards_with_what_we_hold_of_each_paper(client):
    client.put(
        f"/api/reading-list/{PAPER.arxiv_id}",
        json={
            "arxiv_id": PAPER.arxiv_id,
            "version": 6,
            "title": PAPER.title,
            "authors": PAPER.authors,
            "published": "2024-11-23",
        },
    )

    [card] = client.get("/api/papers/search", params={"q": "the LLM judges"}).json()

    assert (card["arxiv_id"], card["indexed"], card["saved"]) == (
        "2411.15594",
        False,
        True,
    )
    # Code writes arXiv's syntax; stopwords are dropped.
    assert client.services.arxiv.queries == ["all:LLM AND all:judges"]


def test_searching_for_an_id_looks_up_that_paper(client):
    [card] = client.get("/api/papers/search", params={"q": "2411.15594v6"}).json()

    assert card["title"] == PAPER.title
    assert client.services.arxiv.queries == []


def test_search_problems_are_clear_errors(client):
    assert client.get("/api/papers/search", params={"q": "the of"}).status_code == 422
    client.services.arxiv.down = True
    response = client.get("/api/papers/search", params={"q": "judges"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_ingest_runs_the_library_and_reports_the_source(client):
    body = client.post("/api/papers/2411.15594/ingest", params={"version": 6}).json()

    assert body["source"] == "full_text"
    assert client.services.library.ingested == [("2411.15594", 6)]
    assert client.post("/api/papers/2499.99999/ingest").status_code == 404


def test_ask_streams_each_step_then_the_answer(client):
    response = client.post(
        "/api/ask",
        json={"question": "Why swap?", "arxiv_id": "2411.15594", "version": 6},
    )

    stream = events(response)
    assert [kind for kind, *_ in stream] == [
        "retrieved",
        "answered",
        "checked",
        "result",
    ]
    assert stream[-1][1]["answer"]["text"] == "Swapping reduces bias [S1]."


def test_ask_about_a_paper_not_yet_indexed_says_so(client):
    response = client.post(
        "/api/ask",
        json={"question": "Why swap?", "arxiv_id": "2499.00001", "version": 1},
    )

    [(kind, data, _)] = events(response)
    assert (kind, data["code"]) == ("error", "not_indexed")


def test_questions_are_limited(client):
    too_long = {"question": "x" * 501, "arxiv_id": "2411.15594", "version": 6}
    not_an_id = {"question": "Why?", "arxiv_id": "drop table", "version": 6}

    assert client.post("/api/ask", json=too_long).status_code == 422
    assert client.post("/api/ask", json=not_an_id).status_code == 422
    assert client.post("/api/reviews", json={"question": "bias?"}).status_code == 422


def test_a_review_is_queued_and_its_events_replay_after_the_last_seen(client):
    jobs = client.services.jobs
    job_id = client.post(
        "/api/reviews", json={"question": "How biased are LLM judges?", "deep": True}
    ).json()["job_id"]
    jobs.add_event(job_id, "step", {"step": "planner", "sub_queries": 3})
    jobs.finish(job_id, {"review": "Done."}, trace_id=None)

    everything = events(client.get(f"/api/reviews/{job_id}/events"))
    resumed = events(
        client.get(f"/api/reviews/{job_id}/events", headers={"Last-Event-ID": "1"})
    )

    assert [kind for kind, *_ in everything] == ["status", "step", "status", "end"]
    assert [event_id for *_, event_id in everything][:3] == ["1", "2", "3"]
    assert everything[1][1]["text"] == "planner: sub_queries 3"
    assert [kind for kind, *_ in resumed] == ["step", "status", "end"]
    view = client.get(f"/api/reviews/{job_id}").json()
    assert (view["job"]["status"], view["result"]) == ("done", {"review": "Done."})


def test_only_a_paused_review_takes_a_decision(client):
    jobs = client.services.jobs
    job_id = jobs.submit("How biased are LLM judges?", {}, pause_for_review=True)
    url = f"/api/reviews/{job_id}/decision"

    assert client.post(url, json={"remove": ["2401.1"]}).status_code == 409
    jobs.claim_next()
    jobs.pause(job_id, {"kept": [], "dropped": [], "question": "q"})
    assert client.get(f"/api/reviews/{job_id}").json()["review_request"]["kept"] == []
    assert client.post(url, json={"remove": ["2401.1"]}).json() == {"status": "queued"}
    assert jobs.get(job_id).decision == {"remove": ["2401.1"], "add": []}


def test_unknown_reviews_are_404_and_bad_ids_422(client):
    missing = "00000000-0000-0000-0000-000000000000"

    assert client.get(f"/api/reviews/{missing}").status_code == 404
    assert client.get(f"/api/reviews/{missing}/events").status_code == 404
    assert client.get(f"/api/reviews/{missing}/graph").status_code == 404
    assert client.get("/api/reviews/not-a-uuid").status_code == 422


def test_the_reading_list_saves_updates_and_removes(client):
    paper = {
        "arxiv_id": "2411.15594",
        "version": 6,
        "title": "A Survey",
        "authors": ["Jiawei Gu"],
        "published": "2024-11-23",
    }
    client.put("/api/reading-list/2411.15594", json=paper)
    client.put("/api/reading-list/2411.15594", json=paper | {"note": "read 3.2"})

    [saved] = client.get("/api/reading-list").json()
    assert saved["note"] == "read 3.2"
    assert client.put("/api/reading-list/2401.00001", json=paper).status_code == 422
    assert client.delete("/api/reading-list/2411.15594").status_code == 204
    assert client.delete("/api/reading-list/2411.15594").status_code == 404


def test_the_citation_graph_comes_from_the_snowball_links():
    result = {
        "kept": [
            {
                "arxiv_id": "A",
                "title": "Kept",
                "via": "search",
                "score": 9,
                "found_by": ["llm judges"],
            },
            {
                "arxiv_id": "B",
                "title": "Snowballed",
                "via": "snowball",
                "score": 8,
                "found_by": ["cited by A"],
            },
        ],
        "dropped": [
            {
                "arxiv_id": "C",
                "title": "Cites A",
                "via": "snowball",
                "score": 3,
                "found_by": ["cites A"],
            },
            {
                "arxiv_id": "D",
                "title": "Unlinked",
                "via": "search",
                "score": 2,
                "found_by": ["bias"],
            },
        ],
        "references": [{"arxiv_id": "B"}],
    }

    graph = citation_graph(result)

    assert [(n.arxiv_id, n.kept, n.cited) for n in graph.nodes] == [
        ("A", True, False),
        ("B", True, True),
        ("C", False, False),
    ]
    assert {(e.source, e.target) for e in graph.edges} == {("A", "B"), ("C", "A")}
