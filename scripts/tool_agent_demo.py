"""Ask a question that an LLM answers by calling MCP tools: a first agent.

    uv run --env-file .env python scripts/tool_agent_demo.py "Which highly cited papers cite MT-Bench (arXiv 2306.05685)?"

The LLM sees the five tools of the arXiv and OpenAlex servers, decides which to
call, reads the results and answers. Every LLM call and tool call is traced in
Langfuse (environment "development"), and the script prints the trace link.
"""

import argparse
import asyncio
import json
import os

from langfuse import get_client

from arxiv_agent.tools.tool_loop import MAX_TOOL_CALLS, run_tool_loop
from arxiv_agent.tools.toolbox import McpToolbox


async def main(question: str, model: str, providers: list[str], budget: int) -> None:
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    langfuse = get_client()
    async with McpToolbox(langfuse=langfuse) as toolbox:
        print("tools:", ", ".join(toolbox.tool_names), "\n")
        result = await run_tool_loop(
            question,
            toolbox,
            model=model,
            providers=providers,
            max_tool_calls=budget,
            langfuse=langfuse,
        )
    langfuse.flush()

    for n, call in enumerate(result.tool_calls, start=1):
        status = "ok" if call.ok else "ERROR"
        arguments = json.dumps(call.arguments, ensure_ascii=False)
        print(f"{n}. {call.tool}({arguments}) -> {status}, {call.duration_ms:.0f} ms")
    print(f"\n{result.answer}\n")
    print(
        f"{result.llm_calls} LLM calls, {len(result.tool_calls)} tool calls, "
        f"stopped by {result.stopped_by}"
    )
    if result.trace_id:
        print(f"trace: {langfuse.get_trace_url(trace_id=result.trace_id)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--providers", default="deepinfra,nscale")
    parser.add_argument("--budget", type=int, default=MAX_TOOL_CALLS)
    args = parser.parse_args()
    asyncio.run(main(args.question, args.model, args.providers.split(","), args.budget))
