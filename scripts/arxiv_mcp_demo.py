"""Call the arXiv MCP server the way an agent will: as a separate process over stdio.

    uv run --env-file .env python scripts/arxiv_mcp_demo.py "position bias LLM judges"
    uv run --env-file .env python scripts/arxiv_mcp_demo.py "LLM-as-a-judge" --category cs.CL

The client launches the server, asks which tools it has, then calls them.
"""

import argparse
import asyncio
import os
import sys

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment

# A stdio server gets only a minimal environment (PATH, TEMP, ...), never your
# secrets. Pass exactly what it needs: the contact details for arXiv's User-Agent.
PASS_THROUGH = ("ARXIV_CONTACT_NAME", "ARXIV_CONTACT_EMAIL")


async def main(query: str, category: str | None, max_results: int) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "arxiv_agent.mcp_servers.arxiv_server"],
        env=get_default_environment()
        | {name: os.environ[name] for name in PASS_THROUGH if name in os.environ},
    )
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        print("tools:", ", ".join(tool.name for tool in tools))

        result = await client.call_tool(
            "search_papers",
            {"query": query, "category": category, "max_results": max_results},
        )
        if result.is_error:
            print("error:", result.content[0].text)
            return
        found = result.structured_content
        print(f"arXiv query: {found['arxiv_query']}\n")
        for paper in found["papers"]:
            authors = ", ".join(paper["authors"])
            if paper["author_count"] > len(paper["authors"]):
                authors += " et al."
            print(f"{paper['arxiv_id']}v{paper['version']}  {paper['published']}")
            print(f"    {paper['title']}\n    {authors}")

        ids = [paper["arxiv_id"] for paper in found["papers"][:2]] + ["2411.99999"]
        lookup = await client.call_tool("get_metadata", {"arxiv_ids": ids})
        metadata = lookup.structured_content
        print(
            f"\nget_metadata: found {[p['arxiv_id'] for p in metadata['papers']]}, "
            f"not found {metadata['not_found']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--category")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.query, args.category, args.max_results))
