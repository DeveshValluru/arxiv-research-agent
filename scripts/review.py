"""Write a short literature review with verified citations.

    uv run --env-file .env python scripts/review.py "How biased are LLMs used as judges?"

Plans searches, finds and screens papers on arXiv, follows their citations,
reads the kept papers, writes a review citing them, and checks every sentence
against its evidence. Needs Postgres running (docker compose up -d) for the
paper index. Traced in Langfuse ("development").
"""

import argparse
import asyncio
import os
from collections import Counter

from langfuse import get_client

from arxiv_agent.clients.arxiv import ArxivClient
from arxiv_agent.evals.judge import JUDGE_MODEL
from arxiv_agent.ingestion.chunker import CHUNK_TOKENIZER
from arxiv_agent.ingestion.embedder import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL_ID,
    Embedder,
    load_token_counter,
)
from arxiv_agent.library import Library
from arxiv_agent.llm import ChatModel
from arxiv_agent.retrieval.retriever import build_retriever
from arxiv_agent.review.critic import Critic
from arxiv_agent.review.graph import build_review_graph, run_review
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.reader import Reader
from arxiv_agent.review.state import Budget, ReviewState
from arxiv_agent.review.writer import Synthesizer
from arxiv_agent.storage.chunk_store import ChunkStore
from arxiv_agent.tools.toolbox import McpToolbox


async def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    langfuse = get_client()
    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    embedder = Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
    library = Library(
        store,
        ArxivClient(),
        embedder,
        build_retriever("rerank", store, embedder),
        load_token_counter(CHUNK_TOKENIZER),
    )
    writer = ChatModel(
        args.model, args.providers.split(","), langfuse=langfuse, timeout=120
    )
    # The judge is from a different model family than the writer. Its calls
    # are short (~2 s), so a provider taking 30 s is treated as down.
    judge = ChatModel(
        JUDGE_MODEL, args.judge_providers.split(","), langfuse=langfuse, timeout=30
    )

    async with McpToolbox(langfuse=langfuse) as toolbox:  # arXiv + OpenAlex
        graph = build_review_graph(
            ReviewNodes(writer, toolbox, keep=args.keep, langfuse=langfuse),
            Reader(writer, library, langfuse=langfuse),
            Synthesizer(writer, langfuse=langfuse),
            Critic(judge, max_revisions=args.max_revisions, langfuse=langfuse),
        )
        budget = Budget(
            max_llm_calls=args.max_llm_calls,
            max_tokens=args.max_tokens,
            max_seconds=args.max_seconds,
        )
        state, trace_id = await run_review(
            args.question, graph, budget=budget, langfuse=langfuse
        )
    store.close()
    langfuse.flush()

    report(state)
    if trace_id:
        print(f"\ntrace: {langfuse.get_trace_url(trace_id=trace_id)}")


def report(state: ReviewState) -> None:
    found = Counter(candidate.via for candidate in state["candidates"])
    print(f"{found['search']} papers from search, {found['snowball']} from snowballing")
    print(f"{len(state['kept'])} kept:")
    for paper in state["kept"]:
        print(f"  {paper.score:2}  {paper.arxiv_id}  [{paper.via}]  {paper.title[:70]}")

    if "read" in state:
        sources = Counter(r.source for r in state["read"])
        rejected = sum(len(r.rejected) for r in state["read"])
        print(
            f"\nRead: {dict(sources)}; {len(state['claims'])} claims "
            f"({rejected} rejected by the quote check)"
        )
    spent = state["spent"]
    print(
        f"Spent: {spent.llm_calls} LLM calls, {spent.tokens:,} tokens, "
        f"at least ${spent.cost_usd:.4f} (not every provider reports a cost)"
    )
    if "stopped" in state:
        print(f"STOPPED EARLY: {state['stopped']}; showing what was done so far")
    if "review" not in state:
        print("\nNo review: nothing relevant was found or read, or time ran out.")
        return

    checks = Counter(c.verdict for c in state["critique"].checks)
    print(
        f"Drafts: {state['drafts']}, final check: {dict(checks)}, "
        f"verdict: {state['critique'].verdict}\n"
    )
    print(state["review"])
    print("\nReferences")
    for paper in state["references"]:
        authors = ", ".join(paper.authors[:2]) + (
            " et al." if len(paper.authors) > 2 else ""
        )
        print(
            f"  [arXiv:{paper.arxiv_id}] {paper.title}. {authors}, {paper.published[:4]}"
        )
    for sentence in state["removed"]:
        print(f"\nREMOVED (still failed the check): {sentence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--providers", default="deepinfra,nscale")
    # Fastest first, measured with 4 parallel calls: together < 1 s, novita
    # ~2 s, ovhcloud ~3 s. (The eval judge keeps its own order, so eval scores
    # stay comparable.)
    parser.add_argument("--judge-providers", default="together,novita,ovhcloud")
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--max-revisions", type=int, default=2)
    defaults = Budget()
    parser.add_argument("--max-llm-calls", type=int, default=defaults.max_llm_calls)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--max-seconds", type=float, default=defaults.max_seconds)
    asyncio.run(main(parser.parse_args()))
