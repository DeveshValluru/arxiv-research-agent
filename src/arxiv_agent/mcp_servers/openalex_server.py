"""OpenAlex as an MCP server: a paper's impact, and the papers that cite it.

    uv run --env-file .env python -m arxiv_agent.mcp_servers.openalex_server

References (what a paper cites) come from the arXiv server's get_references,
read from the paper's own bibliography. Citations (who cites a paper) need a
citation index, which is what this server adds.
"""

import logging
import re
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from arxiv_agent.clients.openalex import (
    OpenAlexClient,
    OpenAlexError,
    OpenAlexUnavailableError,
    Work,
)

MAX_CITATIONS = 25
# Modern arXiv ids (2306.05685) only for now: the arXiv-id matching in
# clients/openalex.py assumes that form, not older ones like hep-th/9901001.
NEW_STYLE_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")


class CitationsResult(BaseModel):
    paper: Work
    total_citations: int
    sort: str
    citing_papers: list[Work]


def _bare_id(arxiv_id: str) -> str:
    match = NEW_STYLE_ID.match(arxiv_id.strip())
    if match is None:
        raise ToolError(
            f"arxiv_id must look like '2306.05685' or '2306.05685v2', got {arxiv_id!r}"
        )
    return match.group(1)


def create_server(client: OpenAlexClient | None = None) -> MCPServer:
    openalex = client or OpenAlexClient()
    server = MCPServer(
        "openalex",
        instructions=(
            "Citation data from OpenAlex, an open index of scholarly works: how "
            "often a paper is cited, and which papers cite it."
        ),
    )

    def find(arxiv_id: str) -> Work:
        bare = _bare_id(arxiv_id)
        try:
            work = openalex.get_work_by_arxiv_id(bare)
        except OpenAlexUnavailableError as exc:
            raise ToolError(
                "OpenAlex is temporarily unavailable. Wait a minute before trying again."
            ) from exc
        except OpenAlexError as exc:
            raise ToolError(str(exc)) from exc
        if work is None:
            raise ToolError(
                f"OpenAlex has no record of arXiv paper {bare}. Very new papers can "
                "take a few days to appear."
            )
        return work

    @server.tool()
    def get_paper(
        arxiv_id: Annotated[
            str, Field(description="arXiv id such as '2306.05685' or '2306.05685v2'.")
        ],
    ) -> Work:
        """Look up a paper's citation count, year and venue by its arXiv id."""
        return find(arxiv_id)

    @server.tool()
    def get_citations(
        arxiv_id: Annotated[
            str, Field(description="arXiv id such as '2306.05685' or '2306.05685v2'.")
        ],
        sort: Annotated[
            Literal["most_cited", "newest"],
            Field(
                description="most_cited finds influential follow-ups; newest finds recent work."
            ),
        ] = "most_cited",
        max_results: Annotated[int, Field(ge=1, le=MAX_CITATIONS)] = 10,
    ) -> CitationsResult:
        """List papers that cite a given paper, with how often each is cited.

        Citing papers have an arxiv_id when they are on arXiv; journal and
        conference papers may only have a DOI.
        """
        paper = find(arxiv_id)
        try:
            total, citing = openalex.get_citations(
                paper.openalex_id, limit=max_results, sort=sort
            )
        except OpenAlexError as exc:
            raise ToolError(f"OpenAlex couldn't list citations: {exc}") from exc
        return CitationsResult(
            paper=paper, total_citations=total, sort=sort, citing_papers=citing
        )

    return server


server = create_server()

if __name__ == "__main__":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    server.run()  # stdio
