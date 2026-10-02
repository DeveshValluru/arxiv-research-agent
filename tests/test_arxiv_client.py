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


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


def ok() -> httpx.Response:
    return httpx.Response(200, text=SEARCH_XML)


def fake_arxiv(*responses: httpx.Response | Exception) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_client(
    clock: FakeClock, *responses: httpx.Response | Exception
) -> ArxivClient:
    return ArxivClient(http=fake_arxiv(*responses), sleep=clock.sleep, clock=clock.time)


def test_search_returns_parsed_papers():
    client = make_client(FakeClock(), ok())

    papers = client.search_papers("anything")

    assert len(papers) == 5
    assert papers[0].arxiv_id == "2411.15594"
    assert papers[0].version == 6


def test_retries_429_with_backoff_then_succeeds():
    clock = FakeClock()
    client = make_client(clock, httpx.Response(429), httpx.Response(429), ok())

    papers = client.search_papers("anything")

    assert len(papers) == 5
    assert clock.waits == [5.0, 10.0]


def test_obeys_retry_after_header():
    clock = FakeClock()
    client = make_client(clock, httpx.Response(429, headers={"Retry-After": "7"}), ok())

    client.search_papers("anything")

    assert clock.waits == [7.0]


def test_timeout_is_retried():
    clock = FakeClock()
    client = make_client(clock, httpx.ReadTimeout("timed out"), ok())

    papers = client.search_papers("anything")

    assert len(papers) == 5
    assert clock.waits == [5.0]


def test_bad_request_raises_query_error_without_retrying():
    clock = FakeClock()
    client = make_client(clock, httpx.Response(400))

    with pytest.raises(ArxivQueryError):
        client.search_papers("anything")
    assert clock.waits == []


def test_gives_up_after_max_attempts():
    clock = FakeClock()
    client = make_client(clock, *[httpx.Response(429) for _ in range(4)])

    with pytest.raises(ArxivUnavailableError):
        client.search_papers("anything")
    assert clock.waits == [5.0, 10.0, 20.0]


def test_paces_back_to_back_requests():
    clock = FakeClock()
    client = make_client(clock, ok(), ok())

    client.search_papers("first")
    client.search_papers("second")

    assert clock.waits == [3.0]


def test_waits_only_the_remaining_time():
    clock = FakeClock()
    client = make_client(clock, ok(), ok())

    client.search_papers("first")
    clock.now += 1.0
    client.search_papers("second")

    assert clock.waits == [2.0]


def test_no_wait_when_enough_time_passed():
    clock = FakeClock()
    client = make_client(clock, ok(), ok())

    client.search_papers("first")
    clock.now += 10.0
    client.search_papers("second")

    assert clock.waits == []
