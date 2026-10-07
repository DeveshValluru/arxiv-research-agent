import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp import Client

from arxiv_agent.clients.arxiv import ArxivUnavailableError, PaperSummary
from arxiv_agent.mcp_servers.arxiv_server import (
    ABSTRACT_CHARS,
    MAX_RESULTS,
    REFERENCE_CHARS,
    build_search_query,
    create_server,
)

FIXTURE_HTML = (Path(__file__).parent / "fixtures" / "latexml_minimal.html").read_text(
    encoding="utf-8"
)


def make_paper(
    arxiv_id: str, authors: int = 2, abstract: str = "Short."
) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        version=2,
        title=f"Paper {arxiv_id}",
        authors=[f"Author {n}" for n in range(authors)],
        abstract=abstract,
        published=datetime(2024, 11, 23, 18, 30, tzinfo=UTC),
        updated=datetime(2025, 1, 2, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
    )


class FakeArxiv:
    def __init__(
        self, papers=(), error: Exception | None = None, html: str | None = None
    ) -> None:
        self.papers = {paper.arxiv_id: paper for paper in papers}
        self.error = error
        self.html = html
        self.searches: list[tuple[str, int]] = []
        self.lookups: list[list[str]] = []
        self.fetches: list[tuple[str, int | None]] = []

    def fetch_html(self, arxiv_id, version=None):
        self.fetches.append((arxiv_id, version))
        if self.error:
            raise self.error
        return self.html

    def search_papers(self, query, max_results=5):
        self.searches.append((query, max_results))
        if self.error:
            raise self.error
        return list(self.papers.values())[:max_results]

    def get_metadata(self, ids):
        self.lookups.append(ids)
        if self.error:
            raise self.error
        bare = [arxiv_id.split("v")[0] for arxiv_id in ids]
        return {
            arxiv_id: self.papers[arxiv_id]
            for arxiv_id in bare
            if arxiv_id in self.papers
        }


def call(arxiv: FakeArxiv, tool: str, arguments: dict, html_cache: Path | None = None):
    async def go():
        async with Client(create_server(arxiv, html_cache)) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(go())


def list_tools():
    async def go():
        async with Client(create_server(FakeArxiv())) as client:
            return (await client.list_tools()).tools

    return asyncio.run(go())


# --- build_search_query: plain words in, arXiv syntax out -------------------


def test_words_are_anded():
    assert build_search_query("position bias judges") == (
        "all:position AND all:bias AND all:judges"
    )


def test_stopwords_are_dropped():
    # Measured: "all:the AND ..." cut 90 relevant results to 9 irrelevant ones.
    assert build_search_query("What is the position bias of LLM judges?") == (
        "all:position AND all:bias AND all:LLM AND all:judges"
    )


def test_quoted_phrases_stay_together():
    assert build_search_query('"LLM as a judge" bias') == (
        'all:"LLM as a judge" AND all:bias'
    )


def test_hyphenated_terms_become_phrases():
    assert build_search_query("LLM-as-a-judge GPT-3.5") == (
        'all:"LLM as a judge" AND all:"GPT 3.5"'
    )


def test_category_is_validated_and_appended():
    assert build_search_query("bias", "cs.CL") == "all:bias AND cat:cs.CL"
    assert build_search_query("bias", "astro-ph.GA") == "all:bias AND cat:astro-ph.GA"
    with pytest.raises(ValueError, match="arXiv category such as 'cs.CL'"):
        build_search_query("bias", "cs CL; DROP")


def test_arxiv_syntax_cannot_be_injected():
    # Colons, parentheses and operators never reach arXiv as syntax.
    assert build_search_query("abs:hack) OR (cat:x") == (
        "all:abs AND all:hack AND all:cat AND all:x"
    )


def test_unclosed_quote_falls_back_to_words():
    assert build_search_query('"position bias') == "all:position AND all:bias"


def test_query_without_keywords_is_rejected():
    with pytest.raises(ValueError, match="at least one keyword"):
        build_search_query('What is the "of the"?')


def test_repeated_terms_appear_once():
    assert build_search_query("bias bias Bias") == "all:bias AND all:Bias"


# --- the MCP tools, called through an in-memory MCP client ------------------


def test_server_offers_two_tools_with_enforced_limits():
    tools = {tool.name: tool for tool in list_tools()}

    assert set(tools) == {"search_papers", "get_metadata", "get_references"}
    max_results = tools["search_papers"].input_schema["properties"]["max_results"]
    assert (max_results["minimum"], max_results["maximum"]) == (1, MAX_RESULTS)
    assert (
        "plain keywords"
        in tools["search_papers"]
        .input_schema["properties"]["query"]["description"]
        .lower()
    )


def test_search_returns_trimmed_papers():
    long_abstract = "word " * 400
    arxiv = FakeArxiv([make_paper("2411.15594", authors=7, abstract=long_abstract)])

    result = call(
        arxiv, "search_papers", {"query": "the position bias", "max_results": 3}
    )

    assert not result.is_error
    assert arxiv.searches == [("all:position AND all:bias", 3)]
    data = result.structured_content
    assert data["arxiv_query"] == "all:position AND all:bias"
    [paper] = data["papers"]
    assert paper["arxiv_id"] == "2411.15594"
    assert paper["authors"] == ["Author 0", "Author 1", "Author 2"]
    assert paper["author_count"] == 7
    assert paper["published"] == "2024-11-23"
    assert len(paper["abstract"]) <= ABSTRACT_CHARS + 2
    assert paper["abstract"].endswith(" …")


def test_bad_category_comes_back_as_an_actionable_error():
    arxiv = FakeArxiv()

    result = call(arxiv, "search_papers", {"query": "bias", "category": "cs CL"})

    assert result.is_error
    assert "arXiv category such as 'cs.CL'" in result.content[0].text
    assert arxiv.searches == []  # never reached arXiv


def test_too_many_results_is_rejected_before_arxiv():
    arxiv = FakeArxiv()
    result = call(arxiv, "search_papers", {"query": "bias", "max_results": 500})
    assert result.is_error
    assert arxiv.searches == []


def test_arxiv_outage_says_to_wait():
    arxiv = FakeArxiv(error=ArxivUnavailableError("down"))
    result = call(arxiv, "search_papers", {"query": "bias"})
    assert result.is_error
    assert "temporarily unavailable" in result.content[0].text


def test_get_metadata_lists_unknown_and_malformed_ids():
    arxiv = FakeArxiv([make_paper("2411.15594")])

    result = call(
        arxiv,
        "get_metadata",
        {"arxiv_ids": ["2411.15594v6", "2411.99999", "not-an-id", "2411.15594v6"]},
    )

    assert not result.is_error
    assert arxiv.lookups == [["2411.15594v6", "2411.99999"]]  # deduped, valid only
    data = result.structured_content
    assert [p["arxiv_id"] for p in data["papers"]] == ["2411.15594"]
    assert data["not_found"] == ["2411.99999", "not-an-id"]


def test_get_metadata_with_only_bad_ids_skips_arxiv():
    arxiv = FakeArxiv()
    result = call(arxiv, "get_metadata", {"arxiv_ids": ["nope"]})
    assert result.structured_content == {"papers": [], "not_found": ["nope"]}
    assert arxiv.lookups == []


def test_get_references_reads_the_bibliography():
    arxiv = FakeArxiv(html=FIXTURE_HTML)

    result = call(arxiv, "get_references", {"arxiv_id": "2499.00001v3"})

    assert not result.is_error
    assert arxiv.fetches == [("2499.00001", 3)]  # bare id + version, as arXiv needs
    data = result.structured_content
    assert (data["total"], data["with_arxiv_id"]) == (3, 2)
    assert [ref["arxiv_id"] for ref in data["references"]] == [
        "2306.05685",
        "2305.18248",
        None,
    ]
    assert all(len(ref["text"]) <= REFERENCE_CHARS + 2 for ref in data["references"])


def test_get_references_can_keep_only_arxiv_papers():
    arxiv = FakeArxiv(html=FIXTURE_HTML)

    result = call(
        arxiv,
        "get_references",
        {"arxiv_id": "2499.00001", "arxiv_only": True, "max_results": 1},
    )

    assert arxiv.fetches == [("2499.00001", None)]
    data = result.structured_content
    assert data["total"] == 3  # the whole bibliography is still counted
    assert [ref["arxiv_id"] for ref in data["references"]] == ["2306.05685"]


def test_get_references_shares_the_html_cache_for_versioned_ids(tmp_path):
    arxiv = FakeArxiv(html=FIXTURE_HTML)

    for _ in range(2):
        result = call(arxiv, "get_references", {"arxiv_id": "2499.00001v3"}, tmp_path)
        assert result.structured_content["total"] == 3
    call(arxiv, "get_references", {"arxiv_id": "2499.00001"}, tmp_path)

    # Fetched once for v3, then read from disk; "latest" always goes to arXiv.
    assert arxiv.fetches == [("2499.00001", 3), ("2499.00001", None)]
    assert (tmp_path / "2499.00001v3.html").exists()


def test_get_references_without_html_points_to_get_metadata():
    result = call(FakeArxiv(html=None), "get_references", {"arxiv_id": "2499.00001"})
    assert result.is_error
    assert "no HTML version" in result.content[0].text
    assert "get_metadata" in result.content[0].text


def test_get_references_rejects_a_malformed_id_before_arxiv():
    arxiv = FakeArxiv(html=FIXTURE_HTML)
    result = call(arxiv, "get_references", {"arxiv_id": "../../etc/passwd"})
    assert result.is_error
    assert arxiv.fetches == []
