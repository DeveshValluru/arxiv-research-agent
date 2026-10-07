import asyncio

from mcp import Client

from arxiv_agent.clients.openalex import OpenAlexUnavailableError, Work
from arxiv_agent.mcp_servers.openalex_server import MAX_CITATIONS, create_server


def make_work(openalex_id: str, arxiv_id: str | None, cited_by: int) -> Work:
    return Work(
        openalex_id=openalex_id,
        title=f"Paper {openalex_id}",
        year=2024,
        cited_by_count=cited_by,
        venue="arXiv (Cornell University)",
        doi=None,
        arxiv_id=arxiv_id,
        authors=["Ada Lovelace"],
        author_count=1,
    )


MT_BENCH = make_work("W4380353763", "2306.05685", 487)
CITING = [make_work("W2", "2309.07430", 800), make_work("W3", None, 634)]


class FakeOpenAlex:
    def __init__(self, works=(MT_BENCH,), error: Exception | None = None) -> None:
        self.works = {work.arxiv_id: work for work in works}
        self.error = error
        self.lookups: list[str] = []
        self.citation_calls: list[tuple] = []

    def get_work_by_arxiv_id(self, arxiv_id):
        self.lookups.append(arxiv_id)
        if self.error:
            raise self.error
        return self.works.get(arxiv_id)

    def get_citations(self, openalex_id, limit=10, sort="most_cited"):
        self.citation_calls.append((openalex_id, limit, sort))
        return 486, CITING[:limit]


def call(openalex: FakeOpenAlex, tool: str, arguments: dict):
    async def go():
        async with Client(create_server(openalex)) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(go())


def list_tools():
    async def go():
        async with Client(create_server(FakeOpenAlex())) as client:
            return (await client.list_tools()).tools

    return asyncio.run(go())


def test_server_offers_two_tools_with_a_fixed_sort_choice():
    tools = {tool.name: tool for tool in list_tools()}

    assert set(tools) == {"get_paper", "get_citations"}
    props = tools["get_citations"].input_schema["properties"]
    assert props["sort"]["enum"] == ["most_cited", "newest"]
    assert props["max_results"]["maximum"] == MAX_CITATIONS


def test_get_paper_strips_the_version():
    openalex = FakeOpenAlex()

    result = call(openalex, "get_paper", {"arxiv_id": "2306.05685v2"})

    assert not result.is_error
    assert openalex.lookups == ["2306.05685"]
    assert result.structured_content["cited_by_count"] == 487


def test_unknown_paper_explains_why():
    result = call(FakeOpenAlex(), "get_paper", {"arxiv_id": "2411.99999"})
    assert result.is_error
    assert "no record of arXiv paper 2411.99999" in result.content[0].text


def test_malformed_id_never_reaches_openalex():
    openalex = FakeOpenAlex()
    result = call(openalex, "get_paper", {"arxiv_id": "https://evil.example/x"})
    assert result.is_error
    assert "must look like '2306.05685'" in result.content[0].text
    assert openalex.lookups == []


def test_get_citations_looks_up_the_paper_then_its_citations():
    openalex = FakeOpenAlex()

    result = call(
        openalex,
        "get_citations",
        {"arxiv_id": "2306.05685", "sort": "newest", "max_results": 2},
    )

    assert not result.is_error
    assert openalex.citation_calls == [("W4380353763", 2, "newest")]
    data = result.structured_content
    assert data["total_citations"] == 486
    assert data["paper"]["openalex_id"] == "W4380353763"
    assert [w["arxiv_id"] for w in data["citing_papers"]] == ["2309.07430", None]
    # "unknown", not a bare null: a missing link doesn't mean "not on arXiv"
    assert [w["on_arxiv"] for w in data["citing_papers"]] == ["yes", "unknown"]


def test_unknown_sort_is_rejected_by_the_schema():
    openalex = FakeOpenAlex()
    result = call(openalex, "get_citations", {"arxiv_id": "2306.05685", "sort": "best"})
    assert result.is_error
    assert openalex.lookups == []


def test_outage_says_to_wait():
    openalex = FakeOpenAlex(error=OpenAlexUnavailableError("down"))
    result = call(openalex, "get_paper", {"arxiv_id": "2306.05685"})
    assert result.is_error
    assert "temporarily unavailable" in result.content[0].text
