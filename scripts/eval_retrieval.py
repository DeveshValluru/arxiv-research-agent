"""Measure retrieval alone on the Q&A eval set: does the evidence reach the top k?

    uv run --env-file .env python scripts/eval_retrieval.py
    uv run --env-file .env python scripts/eval_retrieval.py --methods dense

No LLM and no judge, so a run takes seconds, costs nothing, and gives the same
numbers every time. Unanswerable questions have no evidence to find, and
questions whose evidence never matched our chunks can't be scored, so both are
left out (and counted).
"""

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from arxiv_agent.evals.qa_scoring import gold_chunk_ids
from arxiv_agent.evals.retrieval import (
    first_hit_rank,
    mean_reciprocal_rank,
    recall_at_k,
)
from arxiv_agent.evals.runner import EvalItem, load_items
from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL_ID, Embedder
from arxiv_agent.retrieval.bm25 import BM25Index
from arxiv_agent.retrieval.fusion import reciprocal_rank_fusion
from arxiv_agent.storage.chunk_store import ChunkStore

EVAL_SETS = [Path("evals/qa_qasper.jsonl"), Path("evals/qa_survey.jsonl")]
DEPTH = 20  # how far down each ranking we look
CUTOFFS = (1, 3, 5, 10, 20)

# A search method: question and (arxiv_id, version) in, chunk ids out, best first.
Search = Callable[[str, tuple[str, int]], list[str]]


def make_searches(store: ChunkStore, embedder: Embedder) -> dict[str, Search]:
    def dense(question: str, paper: tuple[str, int]) -> list[str]:
        hits = store.vector_search(
            DEFAULT_MODEL_ID, embedder.embed_query(question), k=DEPTH, papers=[paper]
        )
        return [hit.chunk.chunk_id for hit in hits]

    def keyword(question: str, paper: tuple[str, int]) -> list[str]:
        hits = store.keyword_search(question, k=DEPTH, papers=[paper])
        return [hit.chunk.chunk_id for hit in hits]

    bm25_indexes: dict[tuple[str, int], tuple[list[str], BM25Index]] = {}

    def bm25(question: str, paper: tuple[str, int]) -> list[str]:
        # One index per paper, built on first use: IDF is measured within it.
        if paper not in bm25_indexes:
            chunks = store.get_chunks(*paper)
            bm25_indexes[paper] = (
                [chunk.chunk_id for chunk in chunks],
                BM25Index([chunk.text for chunk in chunks]),
            )
        ids, index = bm25_indexes[paper]
        return [ids[i] for i, _ in index.top(question, DEPTH)]

    def hybrid(question: str, paper: tuple[str, int]) -> list[str]:
        fused = reciprocal_rank_fusion([dense(question, paper), bm25(question, paper)])
        return [chunk_id for chunk_id, _ in fused[:DEPTH]]

    return {"dense": dense, "keyword": keyword, "bm25": bm25, "hybrid": hybrid}


def source_recall(
    ranks: list[int | None], scored: list[tuple[EvalItem, set[str]]], source: str
) -> float:
    picked = [rank for rank, (item, _) in zip(ranks, scored) if item.source == source]
    return recall_at_k(picked, 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", help="default: all of them")
    parser.add_argument("--sets", nargs="+", type=Path, default=EVAL_SETS)
    args = parser.parse_args()

    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    searches = make_searches(
        store, Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
    )
    methods = args.methods or list(searches)

    chunks_by_paper: dict[tuple[str, int], list] = {}
    scored: list[tuple[EvalItem, set[str]]] = []
    skipped = 0
    for item in load_items(args.sets):
        if item.type == "unanswerable":
            continue
        paper = (item.arxiv_id, item.version)
        if paper not in chunks_by_paper:
            chunks_by_paper[paper] = store.get_chunks(*paper)
        gold = gold_chunk_ids(chunks_by_paper[paper], item.evidence)
        if gold:
            scored.append((item, gold))
        else:
            skipped += 1

    ranks: dict[str, list[int | None]] = {}
    for method in methods:
        search = searches[method]
        ranks[method] = [
            first_hit_rank(search(item.question, (item.arxiv_id, item.version)), gold)
            for item, gold in scored
        ]
    store.close()

    print(f"{len(scored)} questions with matched evidence ({skipped} left out)\n")
    header = " ".join(f"{'R@' + str(k):>5}" for k in CUTOFFS)
    print(f"{'method':10} {header}   MRR   R@5 by source")
    sources = sorted({item.source for item, _ in scored})
    for method, method_ranks in ranks.items():
        recalls = " ".join(f"{recall_at_k(method_ranks, k):5.2f}" for k in CUTOFFS)
        by_source = "  ".join(
            f"{source} {source_recall(method_ranks, scored, source):.2f}"
            for source in sources
        )
        print(
            f"{method:10} {recalls}  {mean_reciprocal_rank(method_ranks):.2f}   {by_source}"
        )

    print(f"\nRank of the first gold chunk ('-' = not in the top {DEPTH}):")
    print(f"  {'question':24} " + " ".join(f"{m:>8}" for m in ranks))
    for n, (item, _) in enumerate(scored):
        cells = " ".join(f"{ranks[m][n] or '-':>8}" for m in ranks)
        print(f"  {item.id[:24]:24} {cells}   {item.question[:50]}")


if __name__ == "__main__":
    main()
