from pathlib import Path

import httpx

from arxiv_agent.clients.arxiv import ArxivClient

FIXTURES = Path(__file__).parent / "fixtures"


def fake_arxiv(*responses: httpx.Response) -> httpx.Client:
    queue = list(responses)
    return httpx.Client(transport=httpx.MockTransport(lambda request: queue.pop(0)))


def test_search_returns_parsed_papers():
    xml = (FIXTURES / "arxiv_search_llm_judge.xml").read_text(encoding="utf-8")
    client = ArxivClient(http=fake_arxiv(httpx.Response(200, text=xml)))

    papers = client.search_papers("anything")

    assert len(papers) == 5
    assert papers[0].arxiv_id == "2411.15594"
    assert papers[0].version == 6
