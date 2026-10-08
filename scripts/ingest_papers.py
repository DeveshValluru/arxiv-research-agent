"""Add arXiv papers to the local index.

    uv run --env-file .env python scripts/ingest_papers.py 2411.15594v6 2306.05685
    uv run --env-file .env python scripts/ingest_papers.py --from-evals

A versioned id ingests that version; a bare id asks arXiv for the latest one.
Re-running is cheap: stored metadata, cached HTML, current chunks and existing
vectors are all reused. --from-evals adds every paper the Q&A eval sets ask
about: how CI builds the index the eval gate runs on.
"""

import argparse
import logging
import os
import re

from arxiv_agent.clients.arxiv import ArxivClient, PaperSummary
from arxiv_agent.evals.qa_run import eval_sets
from arxiv_agent.evals.runner import load_items
from arxiv_agent.ingestion.chunker import CHUNK_TOKENIZER
from arxiv_agent.ingestion.embedder import (
    DEFAULT_MODEL_ID,
    Embedder,
    load_token_counter,
)
from arxiv_agent.ingestion.pipeline import ingest_paper
from arxiv_agent.storage.chunk_store import ChunkStore

VERSIONED_ID = re.compile(r"^(?P<id>.+?)(?:v(?P<version>\d+))?$")


def split_id(raw_id: str) -> tuple[str, int | None]:
    match = VERSIONED_ID.match(raw_id)
    version = match.group("version")
    return match.group("id"), int(version) if version else None


def resolve_papers(
    raw_ids: list[str], store: ChunkStore, client: ArxivClient
) -> list[PaperSummary]:
    papers, to_fetch = [], []
    for raw_id in raw_ids:
        arxiv_id, version = split_id(raw_id)
        stored = store.get_paper(arxiv_id, version) if version else None
        if stored:
            papers.append(stored)
        else:
            to_fetch.append(raw_id)

    if to_fetch:
        found = client.get_metadata(to_fetch)
        for raw_id in to_fetch:
            arxiv_id, version = split_id(raw_id)
            paper = found.get(arxiv_id)
            if paper is None:
                print(f"{raw_id}: not found on arXiv, skipped")
            elif version and paper.version != version:
                print(f"{raw_id}: arXiv returned v{paper.version} instead, skipped")
            else:
                papers.append(paper)
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="arXiv ids, e.g. 2411.15594v6")
    parser.add_argument(
        "--from-evals",
        action="store_true",
        help="also every paper in the Q&A eval sets (evals/qa_*.jsonl)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="embedder model id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ids = list(args.ids)
    if args.from_evals:
        items = load_items(eval_sets())
        ids += sorted({f"{item.arxiv_id}v{item.version}" for item in items})
    if not ids:
        parser.error("give arXiv ids, --from-evals, or both")

    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    client = ArxivClient()
    papers = resolve_papers(ids, store, client)
    if not papers:
        return

    embedder = Embedder(args.model)
    count_tokens = load_token_counter(CHUNK_TOKENIZER)
    for paper in papers:
        report = ingest_paper(paper, client, store, embedder, count_tokens)
        print(
            f"{paper.arxiv_id}v{paper.version}: {report.status} | "
            f"chunks written {report.chunks_written} | "
            f"vectors written {report.vectors_written} | {paper.title[:60]}"
        )
    store.close()


if __name__ == "__main__":
    main()
