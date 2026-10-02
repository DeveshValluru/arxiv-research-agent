from pathlib import Path

import httpx
import pytest

from arxiv_agent.clients.arxiv import (
    ArxivClient,
    ArxivQueryError,
    ArxivUnavailableError,
)

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH_XML = (FIXTURES / "arxiv_search_llm_judge.xml").read_text(encoding="utf-8")


def fake_arxiv(*responses: httpx.Response | Exception) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_returns_parsed_papers():
    client = ArxivClient(http=fake_arxiv(httpx.Response(200, text=SEARCH_XML)))

    papers = client.search_papers("anything")

    assert len(papers) == 5
    assert papers[0].arxiv_id == "2411.15594"
    assert papers[0].version == 6


def test_retries_429_with_backoff_then_succeeds():
    waits = []
    client = ArxivClient(
        http=fake_arxiv(
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, text=SEARCH_XML),
        ),
        sleep=waits.append,
    )

    papers = client.search_papers("anything")

    assert len(papers) == 5
    assert waits == [5.0, 10.0]


def test_obeys_retry_after_header():
    waits = []
    client = ArxivClient(
        http=fake_arxiv(
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, text=SEARCH_XML),
        ),
        sleep=waits.append,
    )

    client.search_papers("anything")

    assert waits == [7.0]


def test_timeout_is_retried():
    waits = []
    client = ArxivClient(
        http=fake_arxiv(
            httpx.ReadTimeout("timed out"),
            httpx.Response(200, text=SEARCH_XML),
        ),
        sleep=waits.append,
    )

    papers = client.search_papers("anything")

    assert len(papers) == 5


def test_bad_request_raises_query_error_without_retrying():
    waits = []
    client = ArxivClient(http=fake_arxiv(httpx.Response(400)), sleep=waits.append)

    with pytest.raises(ArxivQueryError):
        client.search_papers("anything")
    assert waits == []


def test_gives_up_after_max_attempts():
    waits = []
    client = ArxivClient(
        http=fake_arxiv(*[httpx.Response(429)] * 4),
        sleep=waits.append,
    )

    with pytest.raises(ArxivUnavailableError):
        client.search_papers("anything")
    assert waits == [5.0, 10.0, 20.0]
