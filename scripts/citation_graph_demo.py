"""Walk one step of the citation graph through two MCP servers at once.

    uv run --env-file .env python scripts/citation_graph_demo.py 2306.05685

The arXiv server reads what the paper cites (its own bibliography); the OpenAlex
server finds who cites it. An agent will combine the two the same way.
"""

import argparse
import asyncio
import os
import sys

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment

# Stdio servers get a minimal environment; pass only the contact details
# (arXiv's User-Agent, OpenAlex's polite pool), never other secrets.
PASS_THROUGH = ("ARXIV_CONTACT_NAME", "ARXIV_CONTACT_EMAIL")


def launch(module: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=get_default_environment()
        | {name: os.environ[name] for name in PASS_THROUGH if name in os.environ},
    )


async def main(arxiv_id: str, shown: int) -> None:
    async with (
        Client(launch("arxiv_agent.mcp_servers.arxiv_server")) as arxiv,
        Client(launch("arxiv_agent.mcp_servers.openalex_server")) as openalex,
    ):
        paper = await openalex.call_tool("get_paper", {"arxiv_id": arxiv_id})
        if paper.is_error:
            print("error:", paper.content[0].text)
            return
        info = paper.structured_content
        print(
            f"{info['title']} ({info['year']}), cited {info['cited_by_count']} times\n"
        )

        refs = await arxiv.call_tool(
            "get_references",
            {"arxiv_id": arxiv_id, "arxiv_only": True, "max_results": shown},
        )
        if refs.is_error:
            print("references:", refs.content[0].text)
        else:
            data = refs.structured_content
            print(
                f"CITES {data['total']} works ({data['with_arxiv_id']} on arXiv); "
                f"the first {len(data['references'])} on arXiv:"
            )
            for ref in data["references"]:
                print(f"  {ref['arxiv_id']}  {ref['text'][:90]}")

        cites = await openalex.call_tool(
            "get_citations", {"arxiv_id": arxiv_id, "max_results": shown}
        )
        if cites.is_error:
            print("\ncitations:", cites.content[0].text)
        else:
            data = cites.structured_content
            print(f"\nCITED BY {data['total_citations']} works; the most cited:")
            for work in data["citing_papers"]:
                where = work["arxiv_id"] or work["doi"] or work["openalex_id"]
                print(f"  {work['cited_by_count']:5}  {where:28}  {work['title'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arxiv_id")
    parser.add_argument("--shown", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.arxiv_id, args.shown))
