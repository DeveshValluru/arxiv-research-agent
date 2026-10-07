"""arXiv as an MCP server: search papers, look them up, and read their bibliographies.

    uv run --env-file .env python -m arxiv_agent.mcp_servers.arxiv_server

That serves over stdio, the way an MCP client launches it. One server process
holds one ArxivClient, so every call through it shares the same 3-second pacing
and retries (arXiv allows one request at a time).
"""

import logging
import re
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from arxiv_agent.clients.arxiv import (
    ARXIV_ID_PATTERN,
    ArxivClient,
    ArxivError,
    ArxivUnavailableError,
    PaperSummary,
)
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.retrieval.bm25 import STOPWORDS

# A quoted phrase, or one word (letters and digits, keeping inner - and .,
# so "LLM-as-a-judge" and "GPT-3.5" stay whole).
TOKEN = re.compile(r'"([^"]*)"|([A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*)')
WORD = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")
CATEGORY = re.compile(r"^[a-z]+(-[a-z]+)?(\.[A-Za-z]+(-[a-z]+)?)?$")
VERSION_SUFFIX = re.compile(r"v\d+$")
MAX_RESULTS = 20
MAX_IDS = 20
ABSTRACT_CHARS = 600
AUTHORS_SHOWN = 3
REFERENCE_CHARS = 200
MAX_REFERENCES = 100


def _phrase(text: str) -> str | None:
    words = WORD.findall(text.replace("-", " "))
    if all(word.lower() in STOPWORDS for word in words):
        return None
    return " ".join(words)


def build_search_query(query: str, category: str | None = None) -> str:
    # The model writes plain words; code writes arXiv's syntax, so the query is
    # always well-formed and the model can't inject operators. Measured on the
    # live API: one stopword ("the") turned a 90-result query into 9 irrelevant
    # papers, so stopwords are dropped; a hyphenated term matched loosely, so
    # it becomes an exact phrase.
    parts: list[str] = []
    for quoted, word in TOKEN.findall(query):
        if quoted or "-" in word:
            phrase = _phrase(quoted or word)
            if phrase:
                parts.append(f'all:"{phrase}"')
        elif word.lower() not in STOPWORDS:
            parts.append(f"all:{word}")
    if not parts:
        raise ValueError(
            "the query needs at least one keyword, e.g. 'position bias LLM judges'"
        )
    if category is not None:
        if not CATEGORY.match(category):
            raise ValueError(
                "category must be an arXiv category such as 'cs.CL' or 'stat.ML', "
                f"got {category!r}"
            )
        parts.append(f"cat:{category}")
    return " AND ".join(dict.fromkeys(parts))


class PaperInfo(BaseModel):
    arxiv_id: str
    version: int
    title: str
    authors: list[str]  # the first few only
    author_count: int
    published: str  # YYYY-MM-DD
    primary_category: str
    abstract: str  # shortened


class SearchResult(BaseModel):
    arxiv_query: str  # what was actually sent to arXiv
    papers: list[PaperInfo]


class MetadataResult(BaseModel):
    papers: list[PaperInfo]
    not_found: list[str]


class Reference(BaseModel):
    text: str  # the bibliography entry, shortened
    arxiv_id: str | None  # set when the entry links to an arXiv paper


class ReferencesResult(BaseModel):
    arxiv_id: str
    total: int  # entries in the bibliography
    with_arxiv_id: int
    references: list[Reference]


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def paper_info(paper: PaperSummary) -> PaperInfo:
    # Trimmed: a model choosing papers needs the gist, not 300-word abstracts
    # and 40-author lists in its context.
    return PaperInfo(
        arxiv_id=paper.arxiv_id,
        version=paper.version,
        title=paper.title,
        authors=paper.authors[:AUTHORS_SHOWN],
        author_count=len(paper.authors),
        published=paper.published.date().isoformat(),
        primary_category=paper.primary_category,
        abstract=shorten(paper.abstract, ABSTRACT_CHARS),
    )


def _unavailable() -> ToolError:
    return ToolError(
        "arXiv is temporarily unavailable (rate limited or down). "
        "Wait a minute before trying again."
    )


def create_server(client: ArxivClient | None = None) -> MCPServer:
    arxiv = client or ArxivClient()
    server = MCPServer(
        "arxiv",
        instructions=(
            "Search arXiv for research papers, look up papers by arXiv id, and "
            "read the list of papers a paper cites. "
            "Results are trimmed: abstracts are shortened and only the first "
            "authors are listed."
        ),
    )

    @server.tool()
    def search_papers(
        query: Annotated[
            str,
            Field(
                description=(
                    "Plain keywords, e.g. 'position bias LLM judges'. Put an exact "
                    "phrase in double quotes: '\"LLM as a judge\" bias'. "
                    "No arXiv query syntax needed."
                )
            ),
        ],
        category: Annotated[
            str | None,
            Field(description="Optional arXiv category, e.g. 'cs.CL' or 'stat.ML'."),
        ] = None,
        max_results: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = 5,
    ) -> SearchResult:
        """Search arXiv by keywords; returns matching papers, most relevant first.

        Use it to discover papers on a topic. Each result has the arXiv id,
        title, first authors, date, category and a shortened abstract. To look
        up papers whose ids you already know, use get_metadata instead.
        """
        try:
            arxiv_query = build_search_query(query, category)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        try:
            papers = arxiv.search_papers(arxiv_query, max_results=max_results)
        except ArxivUnavailableError as exc:
            raise _unavailable() from exc
        except ArxivError as exc:
            raise ToolError(f"arXiv rejected the search: {exc}") from exc
        return SearchResult(
            arxiv_query=arxiv_query, papers=[paper_info(p) for p in papers]
        )

    @server.tool()
    def get_metadata(
        arxiv_ids: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=MAX_IDS,
                description="arXiv ids such as '2411.15594' or '2411.15594v6'.",
            ),
        ],
    ) -> MetadataResult:
        """Look up papers by arXiv id; returns their metadata.

        Ids that arXiv doesn't know, or that aren't valid arXiv ids, are listed
        in not_found instead of failing the whole call.
        """
        requested = list(dict.fromkeys(arxiv_id.strip() for arxiv_id in arxiv_ids))
        valid = [arxiv_id for arxiv_id in requested if ARXIV_ID_PATTERN.match(arxiv_id)]
        try:
            found = arxiv.get_metadata(valid) if valid else {}
        except ArxivUnavailableError as exc:
            raise _unavailable() from exc
        except ArxivError as exc:
            raise ToolError(f"arXiv rejected the lookup: {exc}") from exc

        papers, not_found = [], []
        for arxiv_id in requested:
            paper = found.get(VERSION_SUFFIX.sub("", arxiv_id))
            if paper is None:
                not_found.append(arxiv_id)
            else:
                papers.append(paper_info(paper))
        return MetadataResult(papers=papers, not_found=not_found)

    @server.tool()
    def get_references(
        arxiv_id: Annotated[
            str,
            Field(description="arXiv id such as '2411.15594' or '2411.15594v6'."),
        ],
        arxiv_only: Annotated[
            bool,
            Field(
                description=(
                    "Only return references that have an arXiv id, so they can be "
                    "looked up with get_metadata."
                )
            ),
        ] = False,
        max_results: Annotated[int, Field(ge=1, le=MAX_REFERENCES)] = 30,
    ) -> ReferencesResult:
        """List the papers a paper cites, read from its own bibliography on arXiv.

        Each entry is the shortened bibliography text, plus an arXiv id when the
        cited work is on arXiv. Entries come in the order the bibliography lists
        them (usually alphabetical), not by importance.
        """
        if not ARXIV_ID_PATTERN.match(arxiv_id):
            raise ToolError(
                f"arxiv_id must look like '2411.15594' or '2411.15594v6', "
                f"got {arxiv_id!r}"
            )
        bare = VERSION_SUFFIX.sub("", arxiv_id)
        version = int(arxiv_id[len(bare) + 1 :]) if arxiv_id != bare else None
        try:
            html = arxiv.fetch_html(bare, version)
        except ArxivUnavailableError as exc:
            raise _unavailable() from exc
        except ArxivError as exc:
            raise ToolError(f"arXiv rejected the request: {exc}") from exc
        if html is None:
            raise ToolError(
                f"arXiv has no HTML version of {arxiv_id}, so its bibliography "
                "can't be read. get_metadata still works for it."
            )

        references = parse_arxiv_html(html).references
        picked = [r for r in references if r.arxiv_id] if arxiv_only else references
        return ReferencesResult(
            arxiv_id=arxiv_id,
            total=len(references),
            with_arxiv_id=sum(r.arxiv_id is not None for r in references),
            references=[
                Reference(text=shorten(r.text, REFERENCE_CHARS), arxiv_id=r.arxiv_id)
                for r in picked[:max_results]
            ],
        )

    return server


# The server MCP tools look for ("mcp", "server" or "app" at module level).
server = create_server()

if __name__ == "__main__":
    # httpx logs every request at INFO; keep the server's output to warnings.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    server.run()  # stdio
