"""Compare embedders on the retrieval eval set.

    uv run --env-file .env python scripts/benchmark_embedders.py [model ...]

With no names it runs DEFAULT_MODELS. qwen3-0.6B runs on CPU (~25 min), so it
only runs when named.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from arxiv_agent.evals.retrieval import (
    first_hit_rank,
    gold_indices,
    mean_reciprocal_rank,
    recall_at_k,
)
from arxiv_agent.ingestion.chunker import CHUNK_TOKENIZER, chunk_paper
from arxiv_agent.ingestion.embedder import (
    BGE_QUERY_PREFIX,
    Embedder,
    HostedEmbedder,
    load_token_counter,
)
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.ingestion.models import Chunk

PAGE = Path("data/html/2411.15594v6.html")
EVAL_SET = Path("evals/retrieval_survey.jsonl")
QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)
# Every model embeds the same chunks (the index recipe), so score differences come
# from the embedder alone. MiniLM is gone: its 256-token limit can't hold them.
MODELS: dict[str, Callable[[], Embedder | HostedEmbedder]] = {
    "bge-small": lambda: Embedder(
        "BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX
    ),
    "bge-base": lambda: Embedder(
        "BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY_PREFIX
    ),
    "bge-large": lambda: Embedder(
        "BAAI/bge-large-en-v1.5", query_prefix=BGE_QUERY_PREFIX
    ),
    "qwen3-0.6B": lambda: Embedder(
        "Qwen/Qwen3-Embedding-0.6B", query_prefix=QWEN_QUERY_PROMPT
    ),
    "qwen3-8B-api": lambda: HostedEmbedder(
        "Qwen/Qwen3-Embedding-8B",
        dimension=4096,
        provider="deepinfra",
        query_prefix=QWEN_QUERY_PROMPT,
    ),
}
DEFAULT_MODELS = ["bge-small", "bge-base", "bge-large", "qwen3-8B-api"]


def evaluate(
    embedder: Embedder | HostedEmbedder, chunks: list[Chunk], questions: list[dict]
) -> dict:
    texts = [c.text for c in chunks]

    start = time.perf_counter()
    vectors = embedder.embed_passages([c.embed_text for c in chunks])
    seconds = time.perf_counter() - start

    ranks, query_seconds = [], []
    for question in questions:
        start = time.perf_counter()
        query_vector = embedder.embed_query(question["query"])
        query_seconds.append(time.perf_counter() - start)

        ranking = np.argsort(-(vectors @ query_vector)).tolist()
        gold = gold_indices(texts, question["evidence"])
        ranks.append(first_hit_rank(ranking, gold))
    return {
        "dim": embedder.dimension,
        "seconds": seconds,
        "query_ms": 1000 * float(np.median(query_seconds)),
        "ranks": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", choices=list(MODELS), metavar="model")
    names = parser.parse_args().models or DEFAULT_MODELS

    paper = parse_arxiv_html(PAGE.read_text(encoding="utf-8"))
    chunks = chunk_paper(
        paper, "2411.15594", 6, count_tokens=load_token_counter(CHUNK_TOKENIZER)
    )
    lines = EVAL_SET.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    results = {}
    for name in names:
        print(f"running {name} ...", flush=True)
        results[name] = evaluate(MODELS[name](), chunks, questions)

    print(f"\n{len(chunks)} chunks (index recipe), {len(questions)} questions")
    print(
        f"\n{'model':12} {'dim':>5} {'embed s':>7} {'query ms':>8}  "
        f"{'R@1':>4} {'R@3':>4} {'R@5':>4} {'MRR':>4}"
    )
    for name, r in results.items():
        ranks = r["ranks"]
        print(
            f"{name:12} {r['dim']:5} {r['seconds']:7.1f} {r['query_ms']:8.0f}  "
            f"{recall_at_k(ranks, 1):4.2f} {recall_at_k(ranks, 3):4.2f} "
            f"{recall_at_k(ranks, 5):4.2f} {mean_reciprocal_rank(ranks):4.2f}"
        )

    for kind in ("paraphrase", "exact_term"):
        picked = [i for i, q in enumerate(questions) if q["type"] == kind]
        scores = "  ".join(
            f"{name} {recall_at_k([r['ranks'][i] for i in picked], 5):.2f}"
            for name, r in results.items()
        )
        print(f"\nrecall@5 on {kind} ({len(picked)} questions):  {scores}")

    print("\nRank of the first right chunk per question ('-' = not found):")
    print("  Q  " + " ".join(f"{name:>12}" for name in results))
    for i, question in enumerate(questions):
        cells = " ".join(f"{r['ranks'][i] or '-':>12}" for r in results.values())
        print(f"{i + 1:3}  {cells}   {question['query'][:44]}")


if __name__ == "__main__":
    main()
