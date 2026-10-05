"""Run the retrieval eval set through the stored index.

    uv run --env-file .env python scripts/eval_index.py

Vector search through Postgres must reproduce the in-memory benchmark for
bge-small exactly (R@5 0.80, MRR 0.42): a difference means storage changed
the chunks or the vectors. Keyword search is the new baseline for Phase 3.
"""

import json
import os
from pathlib import Path

from arxiv_agent.evals.retrieval import (
    first_hit_rank,
    gold_indices,
    mean_reciprocal_rank,
    recall_at_k,
)
from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL_ID, Embedder
from arxiv_agent.storage.chunk_store import ChunkStore

PAPER = ("2411.15594", 6)
EVAL_SET = Path("evals/retrieval_survey.jsonl")


def main() -> None:
    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    chunks = store.get_chunks(*PAPER)
    if not chunks:
        raise SystemExit(
            "paper not indexed: run scripts/ingest_papers.py 2411.15594v6 first"
        )
    assert [c.index for c in chunks] == list(range(len(chunks)))
    texts = [c.text for c in chunks]
    lines = EVAL_SET.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    embedder = Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
    searches = {
        "vector": lambda q: store.vector_search(
            DEFAULT_MODEL_ID, embedder.embed_query(q), k=len(chunks), papers=[PAPER]
        ),
        "keyword": lambda q: store.keyword_search(q, k=len(chunks), papers=[PAPER]),
    }

    results = {}
    for name, search in searches.items():
        ranks = []
        for question in questions:
            ranking = [hit.chunk.index for hit in search(question["query"])]
            gold = gold_indices(texts, question["evidence"])
            ranks.append(first_hit_rank(ranking, gold))
        results[name] = ranks

    print(f"{len(chunks)} chunks, {len(questions)} questions\n")
    print(f"{'search':8} {'R@1':>4} {'R@3':>4} {'R@5':>4} {'MRR':>4}   R@5 by type")
    for name, ranks in results.items():
        by_type = "  ".join(
            f"{kind} {recall_at_k([r for r, q in zip(ranks, questions) if q['type'] == kind], 5):.2f}"
            for kind in ("paraphrase", "exact_term")
        )
        print(
            f"{name:8} {recall_at_k(ranks, 1):4.2f} {recall_at_k(ranks, 3):4.2f} "
            f"{recall_at_k(ranks, 5):4.2f} {mean_reciprocal_rank(ranks):4.2f}   {by_type}"
        )

    print("\nRank of the first right chunk per question ('-' = not found):")
    print(f"  Q  {'vector':>7} {'keyword':>7}")
    for i, question in enumerate(questions):
        cells = " ".join(f"{ranks[i] or '-':>7}" for ranks in results.values())
        print(f"{i + 1:3}  {cells}   {question['type'][:5]}  {question['query'][:50]}")
    store.close()


if __name__ == "__main__":
    main()
