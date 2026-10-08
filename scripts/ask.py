"""Ask a question about one indexed paper.

    uv run --env-file .env python scripts/ask.py 2411.15594v6 "What is position bias?"

The paper must be ingested first (scripts/ingest_papers.py). Add --json to see the
full QAResult record: every source, score, token count and version. Each run is
traced to Langfuse (environment "development") and prints a link to its trace.
"""

import argparse
import os
import re

from langfuse import get_client

from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL_ID, Embedder
from arxiv_agent.qa.answerer import Answerer
from arxiv_agent.qa.support import FAILING, SupportChecker
from arxiv_agent.retrieval.retriever import build_retriever
from arxiv_agent.storage.chunk_store import ChunkStore

VERSIONED_ID = re.compile(r"^(?P<id>.+)v(?P<version>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paper", help="versioned arXiv id, e.g. 2411.15594v6")
    parser.add_argument("question")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--providers",
        default="deepinfra,nscale",
        help="comma-separated, tried in order when one is busy or down",
    )
    parser.add_argument("-k", type=int, default=5, help="sources sent to the model")
    parser.add_argument(
        "--retriever", choices=["dense", "hybrid", "rerank"], default="rerank"
    )
    parser.add_argument("--json", action="store_true", help="print the full record")
    parser.add_argument(
        "--no-support-check",
        action="store_true",
        help="skip checking each sentence against its sources",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="remove failing sentences without asking for a rewrite first",
    )
    args = parser.parse_args()

    match = VERSIONED_ID.match(args.paper)
    if match is None:
        parser.error(f"expected a versioned id like 2411.15594v6, got {args.paper!r}")
    arxiv_id, version = match.group("id"), int(match.group("version"))

    # The application, not the library, decides which environment traces belong to.
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    langfuse = get_client()

    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    embedder = Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
    answerer = Answerer(
        build_retriever(args.retriever, store, embedder),
        model=args.model,
        providers=args.providers.split(","),
        k=args.k,
        langfuse=langfuse,
        support=None if args.no_support_check else SupportChecker(langfuse=langfuse),
        repair=not args.no_repair,
    )
    result = answerer.ask(args.question, arxiv_id, version)
    store.close()
    langfuse.flush()

    if args.json:
        print(result.model_dump_json(indent=2))
        return

    print(f"\n{result.answer.text}\n")
    print(f"[{result.answer.status}]", "; ".join(result.answer.problems))
    checked = [s for s in result.support if s.verdict != "uncited"]
    if checked:
        failed = sum(s.verdict in FAILING for s in checked)
        print(
            f"support check: {len(checked) - failed} of {len(checked)} cited "
            f"sentences supported ({result.support_ms:.0f} ms)"
        )
    if result.repair is not None:
        why = f": {result.repair.reason}" if result.repair.reason else ""
        print(f"repair: {result.repair.outcome}{why}")
    print("\nSources:")
    for source in result.sources:
        cited = "*" if int(source.label[1:]) in result.answer.cited else " "
        path = " > ".join(source.section_path)
        print(f" {cited} {source.label} {source.score:.3f}  {path}")
    cost = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "cost unknown"
    print(
        f"\n{result.model} via {result.provider} ({result.llm_attempts} attempt(s)) | "
        f"tokens {result.prompt_tokens} in + {result.completion_tokens} out | {cost} | "
        f"retrieval {result.retrieval_ms:.0f} ms, generation {result.generation_ms:.0f} ms"
    )
    if result.trace_id:
        print(f"trace: {langfuse.get_trace_url(trace_id=result.trace_id)}")


if __name__ == "__main__":
    main()
