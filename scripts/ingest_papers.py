"""Add arXiv papers to the local index.

    uv run --env-file .env python scripts/ingest_papers.py 2411.15594v6 2306.05685

A versioned id ingests that version; a bare id asks arXiv for the latest one.
Re-running is cheap: stored metadata, cached HTML, current chunks and existing
vectors are all reused.
"""

import argparse
import logging
import os
import re

from arxiv_agent.clients.arxiv import ArxivClient, PaperSummary
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
    parser.add_argument("ids", nargs="+", help="arXiv ids, e.g. 2411.15594v6")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="embedder model id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    client = ArxivClient()
    papers = resolve_papers(args.ids, store, client)
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
