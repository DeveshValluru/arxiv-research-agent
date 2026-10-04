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
IDLIST_XML = (FIXTURES / "arxiv_idlist_found_and_unknown.xml").read_text(
    encoding="utf-8"
)
MALFORMED_400_XML = (FIXTURES / "arxiv_idlist_malformed_400.xml").read_text(
    encoding="utf-8"
)
EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>no results</title></feed>"""


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


def empty() -> httpx.Response:
    return httpx.Response(200, text=EMPTY_FEED)


def fake_arxiv(
    *responses: httpx.Response | Exception,
    seen: list[httpx.Request] | None = None,
) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_client(
    clock: FakeClock,
    *responses: httpx.Response | Exception,
    seen: list[httpx.Request] | None = None,
) -> ArxivClient:
    return ArxivClient(
        http=fake_arxiv(*responses, seen=seen), sleep=clock.sleep, clock=clock.time
    )


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


def test_retry_is_logged(caplog):
    clock = FakeClock()
    client = make_client(clock, httpx.Response(429), ok())

    client.search_papers("anything")

    assert "arXiv HTTP 429 on attempt 1/4; retrying in 5 s" in caplog.text


def test_get_metadata_omits_unknown_ids():
    seen = []
    client = make_client(FakeClock(), httpx.Response(200, text=IDLIST_XML), seen=seen)

    found = client.get_metadata(["2411.15594", "2499.99999", "2411.15594"])

    assert set(found) == {"2411.15594"}
    assert found["2411.15594"].title == "A Survey on LLM-as-a-Judge"
    assert len(seen) == 1
    assert seen[0].url.params["id_list"] == "2411.15594,2499.99999"


def test_get_metadata_never_sends_malformed_ids():
    seen = []
    client = make_client(FakeClock(), seen=seen)

    found = client.get_metadata(["not-an-id", "24ll.1559"])

    assert found == {}
    assert seen == []


def test_get_metadata_batches_and_sets_max_results():
    seen = []
    ids = [f"2401.{n:05d}" for n in range(120)]
    client = make_client(FakeClock(), empty(), empty(), empty(), seen=seen)

    client.get_metadata(ids)

    assert [len(r.url.params["id_list"].split(",")) for r in seen] == [50, 50, 20]
    assert [r.url.params["max_results"] for r in seen] == ["50", "50", "20"]


def test_bad_request_includes_arxiv_explanation():
    client = make_client(FakeClock(), httpx.Response(400, text=MALFORMED_400_XML))

    with pytest.raises(ArxivQueryError, match="incorrect id format for not-an-id"):
        client.search_papers("anything")


def test_fetch_html_returns_page():
    seen = []
    page = "<html><body><h1>A Survey on LLM-as-a-Judge</h1></body></html>"
    client = make_client(FakeClock(), httpx.Response(200, text=page), seen=seen)

    html = client.fetch_html("2411.15594", version=6)

    assert html == page
    assert str(seen[0].url) == "https://arxiv.org/html/2411.15594v6"


def test_fetch_html_without_version_asks_for_latest():
    seen = []
    client = make_client(FakeClock(), httpx.Response(200, text="<html/>"), seen=seen)

    client.fetch_html("2411.15594")

    assert str(seen[0].url) == "https://arxiv.org/html/2411.15594"


def test_fetch_html_returns_none_when_missing():
    clock = FakeClock()
    seen = []
    client = make_client(clock, httpx.Response(404), seen=seen)

    assert client.fetch_html("2411.15594", version=6) is None
    assert len(seen) == 1
    assert clock.waits == []


@pytest.mark.parametrize("bad_id", ["2411.15594v6", "not-an-id", " 2411.15594"])
def test_fetch_html_rejects_non_bare_ids(bad_id):
    seen = []
    client = make_client(FakeClock(), seen=seen)

    with pytest.raises(ValueError):
        client.fetch_html(bad_id)
    assert seen == []
