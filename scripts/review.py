"""Run a quick literature review: plan, search arXiv, screen the abstracts.

    uv run --env-file .env python scripts/review.py "How biased are LLMs used as judges?"

5.1a stops after screening: you get the plan, every candidate, and which papers
were kept or dropped with the reason. Traced in Langfuse ("development").
"""

import argparse
import asyncio
import os

from langfuse import get_client

from arxiv_agent.llm import ChatModel
from arxiv_agent.review.graph import run_review
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.tools.toolbox import SERVERS, McpToolbox, stdio_server


async def main(question: str, model: str, providers: list[str], keep: int) -> None:
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    langfuse = get_client()
    # Only the arXiv server is needed until 5.1b adds snowballing and checks.
    servers = {"arxiv": stdio_server(SERVERS["arxiv"])}
    async with McpToolbox(servers, langfuse=langfuse) as toolbox:
        nodes = ReviewNodes(
            ChatModel(model, providers, langfuse=langfuse),
            toolbox,
            keep=keep,
            langfuse=langfuse,
        )
        state, trace_id = await run_review(question, nodes, langfuse=langfuse)
    langfuse.flush()

    print("PLAN")
    for query in state["sub_queries"]:
        print(f"  search: {query}")
    for criterion in state["criteria"]:
        print(f"  keep if: {criterion}")
    print(f"\n{len(state['candidates'])} candidates, {len(state['kept'])} kept\n")
    for paper in state["kept"]:
        print(f"KEEP {paper.score:2}  {paper.arxiv_id}  {paper.title[:70]}")
        print(f"          {paper.reason}")
    print()
    for paper in state["dropped"]:
        print(f"drop {paper.score:2}  {paper.arxiv_id}  {paper.title[:60]}")
        print(f"          {paper.reason}")
    if trace_id:
        print(f"\ntrace: {langfuse.get_trace_url(trace_id=trace_id)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--providers", default="deepinfra,nscale")
    parser.add_argument("--keep", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.question, args.model, args.providers.split(","), args.keep))
