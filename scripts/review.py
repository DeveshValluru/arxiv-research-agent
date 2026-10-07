"""Write a short literature review with verified citations, in the foreground.

    uv run --env-file .env python scripts/review.py "How biased are LLMs used as judges?"

Plans searches, finds and screens papers on arXiv, follows their citations,
reads the kept papers, writes a review citing them, and checks every sentence
against its evidence, printing progress as it goes. Needs Postgres running
(docker compose up -d). Traced in Langfuse ("development").

With --pause it stops after screening and asks you which papers to remove or
add before anything is read. To run a review in the background instead, see
scripts/review_jobs.py.
"""

import argparse
import asyncio
import os
import time
import uuid

from langfuse import get_client

from arxiv_agent.review.checkpoints import memory_checkpointer
from arxiv_agent.review.events import (
    describe_event,
    format_report,
    format_review_request,
    result_of,
)
from arxiv_agent.review.graph import run_review
from arxiv_agent.review.setup import ReviewSettings, open_review_graph
from arxiv_agent.review.state import Budget


async def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    langfuse = get_client()
    settings = ReviewSettings(
        model=args.model,
        providers=args.providers.split(","),
        judge_providers=args.judge_providers.split(","),
        keep=args.keep,
        max_revisions=args.max_revisions,
    )
    budget = Budget(
        max_llm_calls=args.max_llm_calls,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
    )
    start = time.monotonic()

    def show(kind: str, data: dict) -> None:
        print(f"[{time.monotonic() - start:4.0f}s] {describe_event(kind, data)}")

    # In memory: this review lives as long as the script. Jobs use Postgres.
    async with open_review_graph(settings, langfuse, memory_checkpointer()) as graph:
        thread_id = str(uuid.uuid4())
        run = await run_review(
            args.question,
            graph,
            thread_id=thread_id,
            budget=budget,
            pause_for_review=args.pause,
            on_event=show,
            langfuse=langfuse,
        )
        if run.pause is not None:
            print("\n" + format_review_request(run.pause))
            decision = ask_for_decision()
            run = await run_review(
                args.question,
                graph,
                thread_id=thread_id,
                decision=decision,
                on_event=show,
                langfuse=langfuse,
            )
    langfuse.flush()

    print("\n" + format_report(result_of(run.state)))
    if run.trace_id:
        print(f"\ntrace: {langfuse.get_trace_url(trace_id=run.trace_id)}")


def ask_for_decision() -> dict:
    print(
        "\nEdit the list before it's read: arXiv ids separated by spaces, Enter for none."
    )
    remove = input("Remove: ").split()
    add = input("Add: ").split()
    return {"remove": remove, "add": add}


if __name__ == "__main__":
    defaults, budget = ReviewSettings(), Budget()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--pause", action="store_true", help="review the paper list before reading"
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--providers", default=",".join(defaults.providers))
    parser.add_argument("--judge-providers", default=",".join(defaults.judge_providers))
    parser.add_argument("--keep", type=int, default=defaults.keep)
    parser.add_argument("--max-revisions", type=int, default=defaults.max_revisions)
    parser.add_argument("--max-llm-calls", type=int, default=budget.max_llm_calls)
    parser.add_argument("--max-tokens", type=int, default=budget.max_tokens)
    parser.add_argument("--max-seconds", type=float, default=budget.max_seconds)
    asyncio.run(main(parser.parse_args()))
