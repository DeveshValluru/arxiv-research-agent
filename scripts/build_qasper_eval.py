"""Build the QASPER part of the Q&A eval set.

    uv run --env-file .env python scripts/build_qasper_eval.py --papers 8

Walks QASPER's test split in order and keeps papers with at least one question
that every annotator marked unanswerable. Only about 4% of questions are, so a
plain sample would leave almost nothing to test refusals on. Each kept paper is
ingested through our own pipeline, its questions are converted to our format,
and the result goes to evals/qa_qasper.jsonl. The report says how many answerable
questions found their evidence in our chunks: the rest stay in for answer
grading but are left out of evidence recall.
"""

import argparse
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import httpx

from arxiv_agent.clients.arxiv import shared_arxiv_client
from arxiv_agent.evals.retrieval import match_evidence
from arxiv_agent.ingestion.chunker import CHUNK_TOKENIZER
from arxiv_agent.ingestion.embedder import (
    DEFAULT_MODEL_ID,
    Embedder,
    load_token_counter,
)
from arxiv_agent.ingestion.pipeline import ingest_paper
from arxiv_agent.storage.chunk_store import ChunkStore

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 40
TEST_SPLIT_ROWS = 416
OUTPUT = Path("evals/qa_qasper.jsonl")


def qasper_rows() -> Iterator[dict]:
    # Hugging Face's dataset viewer serves the rows as JSON, page by page.
    for offset in range(0, TEST_SPLIT_ROWS, PAGE_SIZE):
        response = httpx.get(
            ROWS_URL,
            params={
                "dataset": "allenai/qasper",
                "config": "qasper",
                "split": "test",
                "offset": offset,
                "length": PAGE_SIZE,
            },
            timeout=120,
        )
        response.raise_for_status()
        yield from (item["row"] for item in response.json()["rows"])


def is_unanswerable(answers: dict) -> bool:
    return all(a["unanswerable"] for a in answers["answer"])


def answer_text(annotation: dict) -> str:
    if annotation["yes_no"] is not None:
        return "Yes" if annotation["yes_no"] else "No"
    if annotation["free_form_answer"].strip():
        return annotation["free_form_answer"].strip()
    return "; ".join(span.strip() for span in annotation["extractive_spans"])


def answer_type(annotation: dict) -> str:
    if annotation["yes_no"] is not None:
        return "yes_no"
    if annotation["free_form_answer"].strip():
        return "abstractive"
    return "extractive"


def convert(row: dict, version: int) -> list[dict]:
    qas = row["qas"]
    items = []
    for question, question_id, answers in zip(
        qas["question"], qas["question_id"], qas["answers"]
    ):
        annotations = answers["answer"]
        flags = [a["unanswerable"] for a in annotations]
        if any(flags) and not all(flags):
            continue  # annotators disagree on whether it's answerable: no clear gold

        gold, evidence = [], []
        if not all(flags):
            gold = list(dict.fromkeys(g for g in map(answer_text, annotations) if g))
            if not gold:
                continue
            evidence = list(
                dict.fromkeys(
                    e
                    for a in annotations
                    for e in a["evidence"]
                    if not e.startswith("FLOAT SELECTED")  # figure/table captions
                )
            )
        items.append(
            {
                "id": f"qasper-{question_id}",
                "source": "qasper",
                "arxiv_id": row["id"],
                "version": version,
                "question": question.strip(),
                "type": "unanswerable" if all(flags) else "answerable",
                "answer_type": None if all(flags) else answer_type(annotations[0]),
                "gold_answers": gold,
                "evidence": evidence,
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--papers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    client = shared_arxiv_client()
    embedder = Embedder(DEFAULT_MODEL_ID)
    count_tokens = load_token_counter(CHUNK_TOKENIZER)

    items: list[dict] = []
    kept = 0
    for row in qasper_rows():
        if not any(is_unanswerable(a) for a in row["qas"]["answers"]):
            continue
        paper = client.get_metadata([row["id"]]).get(row["id"])
        if paper is None:
            print(f"{row['id']}: not found on arXiv, skipped")
            continue
        report = ingest_paper(paper, client, store, embedder, count_tokens)
        if report.status != "ingested":
            print(f"{row['id']}: no HTML version, skipped")
            continue

        chunk_texts = [c.text for c in store.get_chunks(paper.arxiv_id, paper.version)]
        paper_items = convert(row, paper.version)
        for item in paper_items:
            item["evidence_matched"] = bool(
                match_evidence(chunk_texts, item["evidence"])
            )
        items.extend(paper_items)
        kept += 1
        print(
            f"{paper.arxiv_id}v{paper.version}: {len(paper_items)} questions | "
            f"chunks written {report.chunks_written} | {paper.title[:50]}"
        )
        if kept == args.papers:
            break
    store.close()

    OUTPUT.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
        newline="\n",
    )
    answerable = [i for i in items if i["type"] == "answerable"]
    matched = sum(i["evidence_matched"] for i in answerable)
    print(f"\n{len(items)} questions from {kept} papers -> {OUTPUT}")
    print(
        f"  answerable {len(answerable)}, unanswerable {len(items) - len(answerable)}"
    )
    print(f"  evidence found in our chunks: {matched}/{len(answerable)} answerable")


if __name__ == "__main__":
    main()
