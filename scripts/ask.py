"""Ask a question about one indexed paper.

    uv run --env-file .env python scripts/ask.py 2411.15594v6 "What is position bias?"

The paper must be ingested first (scripts/ingest_papers.py). Add --json to see the
full QAResult record: every source, score, token count and version.
"""

import argparse
import os
import re

from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL_ID, Embedder
from arxiv_agent.qa.answerer import Answerer
from arxiv_agent.storage.chunk_store import ChunkStore

VERSIONED_ID = re.compile(r"^(?P<id>.+)v(?P<version>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paper", help="versioned arXiv id, e.g. 2411.15594v6")
    parser.add_argument("question")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--provider", default="deepinfra")
    parser.add_argument("-k", type=int, default=5, help="sources sent to the model")
    parser.add_argument("--json", action="store_true", help="print the full record")
    args = parser.parse_args()

    match = VERSIONED_ID.match(args.paper)
    if match is None:
        parser.error(f"expected a versioned id like 2411.15594v6, got {args.paper!r}")
    arxiv_id, version = match.group("id"), int(match.group("version"))

    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    answerer = Answerer(
        store,
        Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX),
        model=args.model,
        provider=args.provider,
        k=args.k,
    )
    result = answerer.ask(args.question, arxiv_id, version)
    store.close()

    if args.json:
        print(result.model_dump_json(indent=2))
        return

    print(f"\n{result.answer.text}\n")
    print(f"[{result.answer.status}]", "; ".join(result.answer.problems))
    print("\nSources:")
    for source in result.sources:
        cited = "*" if int(source.label[1:]) in result.answer.cited else " "
        print(
            f" {cited} {source.label} {source.score:.3f}  {' > '.join(source.section_path)}"
        )
    cost = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "cost unknown"
    print(
        f"\n{result.model} via {result.provider} | "
        f"tokens {result.prompt_tokens} in + {result.completion_tokens} out | {cost} | "
        f"retrieval {result.retrieval_ms:.0f} ms, generation {result.generation_ms:.0f} ms"
    )


if __name__ == "__main__":
    main()
