import itertools

import httpx
import pytest

from arxiv_agent.clients.openalex import (
    OpenAlexClient,
    OpenAlexError,
    OpenAlexUnavailableError,
    to_work,
)

MT_BENCH = {
    "id": "https://openalex.org/W4380353763",
    "doi": "https://doi.org/10.48550/arxiv.2306.05685",
    "display_name": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
    "publication_year": 2023,
    "cited_by_count": 487,
    "primary_location": {"source": {"display_name": "arXiv (Cornell University)"}},
    "locations": [{"landing_page_url": "http://arxiv.org/abs/2306.05685"}],
    "authorships": [
        {"author": {"display_name": name}}
        for name in ("Lianmin Zheng", "Wei-Lin Chiang", "Ying Sheng", "Siyuan Zhuang")
    ],
}
G_EVAL = {  # a conference version: journal DOI, no arXiv copy listed
    "id": "https://openalex.org/W4389520749",
    "doi": "https://doi.org/10.18653/v1/2023.emnlp-main.153",
    "display_name": "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment",
    "publication_year": 2023,
    "cited_by_count": 801,
    "primary_location": {"source": {"display_name": "EMNLP 2023"}},
    "locations": [{"landing_page_url": "https://aclanthology.org/2023.emnlp-main.153"}],
    "authorships": [{"author": {"display_name": "Yang Liu"}}],
}
MEDICAL = {  # a journal DOI, but one copy lives on arXiv
    "id": "https://openalex.org/W4392312047",
    "doi": "https://doi.org/10.1038/s41591-024-02855-5",
    "display_name": "Adapted large language models can outperform medical experts",
    "publication_year": 2024,
    "cited_by_count": 800,
    "primary_location": {"source": {"display_name": "Nature Medicine"}},
    "locations": [
        {"landing_page_url": "https://www.nature.com/articles/s41591-024-02855-5"},
        {"landing_page_url": "https://arxiv.org/abs/2309.07430v5"},
    ],
    "authorships": [],
}


def make_client(handler, sleeps=None) -> OpenAlexClient:
    ticks = itertools.count(step=10)  # a clock that moves fast: no pacing waits
    return OpenAlexClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        mailto="dev@example.com",
        sleep=(sleeps if sleeps is not None else []).append,
        clock=lambda: next(ticks),
    )


def test_to_work_trims_and_normalizes():
    work = to_work(MT_BENCH)

    assert work.openalex_id == "W4380353763"
    assert work.doi == "10.48550/arxiv.2306.05685"  # an identifier, not a URL
    assert work.arxiv_id == "2306.05685"
    assert (work.year, work.cited_by_count) == (2023, 487)
    assert work.venue == "arXiv (Cornell University)"
    assert work.authors == ["Lianmin Zheng", "Wei-Lin Chiang", "Ying Sheng"]
    assert work.author_count == 4


def test_arxiv_id_comes_from_a_location_when_the_doi_is_a_journal_doi():
    assert to_work(MEDICAL).arxiv_id == "2309.07430"


def test_works_not_on_arxiv_have_no_arxiv_id():
    work = to_work(G_EVAL)
    assert work.arxiv_id is None
    assert work.doi == "10.18653/v1/2023.emnlp-main.153"


def test_lookup_by_arxiv_id_uses_the_arxiv_doi_and_the_polite_pool():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=MT_BENCH)

    work = make_client(handler).get_work_by_arxiv_id("2306.05685")

    assert work.title.startswith("Judging LLM-as-a-Judge")
    [request] = seen
    assert request.url.path == "/works/doi:10.48550/arXiv.2306.05685"
    assert request.url.params["mailto"] == "dev@example.com"
    assert "cited_by_count" in request.url.params["select"]


def test_unknown_paper_is_none():
    client = make_client(lambda request: httpx.Response(404, json={}))
    assert client.get_work_by_arxiv_id("2411.99999") is None


def test_citations_are_filtered_sorted_and_counted():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, json={"meta": {"count": 486}, "results": [G_EVAL, MEDICAL]}
        )

    total, works = make_client(handler).get_citations("W4380353763", limit=2)

    assert total == 486
    assert [w.openalex_id for w in works] == ["W4389520749", "W4392312047"]
    params = seen[0].url.params
    assert params["filter"] == "cites:W4380353763"
    assert params["sort"] == "cited_by_count:desc"
    assert params["per-page"] == "2"


def test_newest_sorts_by_publication_date():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"meta": {"count": 0}, "results": []})

    make_client(handler).get_citations("W1", sort="newest")
    assert seen[0].url.params["sort"] == "publication_date:desc"


def test_rate_limit_is_retried_with_backoff():
    responses = iter([httpx.Response(429), httpx.Response(200, json=MT_BENCH)])
    sleeps = []

    work = make_client(lambda request: next(responses), sleeps).get_work_by_arxiv_id(
        "2306.05685"
    )

    assert work.openalex_id == "W4380353763"
    assert sleeps == [1.0]


def test_a_bad_request_is_not_retried():
    sleeps = []
    client = make_client(lambda request: httpx.Response(400), sleeps)
    with pytest.raises(OpenAlexError, match="HTTP 400"):
        client.get_work_by_arxiv_id("2306.05685")
    assert sleeps == []


def test_persistent_outage_gives_up():
    client = make_client(lambda request: httpx.Response(503))
    with pytest.raises(OpenAlexUnavailableError, match="after 4 attempts"):
        client.get_work_by_arxiv_id("2306.05685")
