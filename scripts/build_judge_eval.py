"""Build the judge eval cases from the local paper index.

    uv run --env-file .env python scripts/build_judge_eval.py

Picks real sentences that state a number from the indexed papers (one per
chunk, two per paper, papers in a fixed order) and breaks copies of them in
known ways (see arxiv_agent/evals/judge_eval.py). The cases quote paper text,
which not every arXiv license lets us redistribute, so they go to
data/evals/judge_support.jsonl, not to git. Same index, same cases.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from arxiv_agent.evals.judge_eval import Base, build_cases, candidate_sentences

OUTPUT = Path("data/evals/judge_support.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-paper", type=int, default=2)
    parser.add_argument("--bases", type=int, default=40, help="real sentences to use")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT arxiv_id, chunk_id, section_path, text FROM chunks "
            "WHERE kind = 'text' ORDER BY arxiv_id, chunk_index"
        ).fetchall()

    bases: list[Base] = []
    per_paper: Counter = Counter()
    for row in rows:
        if per_paper[row["arxiv_id"]] >= args.per_paper:
            continue
        sentences = candidate_sentences(row["text"])
        if not sentences:
            continue
        bases.append(
            Base(
                arxiv_id=row["arxiv_id"],
                chunk_id=row["chunk_id"],
                section=" > ".join(row["section_path"]),
                sentence=sentences[0],
                passage=row["text"],
            )
        )
        per_paper[row["arxiv_id"]] += 1
        if len(bases) == args.bases:
            break

    cases = build_cases(bases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    kinds = Counter(case.kind for case in cases)
    print(
        f"{len(bases)} sentences from {len(per_paper)} papers -> {len(cases)} cases "
        f"{dict(kinds)}\nwrote {args.out}"
    )


if __name__ == "__main__":
    main()
