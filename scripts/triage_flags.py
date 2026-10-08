"""Turn harvested flags into eval cases, one decision at a time (7.3).

    uv run --env-file .env python scripts/triage_flags.py
    uv run --env-file .env python scripts/triage_flags.py --list

For each pending candidate in data/evals/flag_candidates.jsonl (from
harvest_flags.py) it shows what happened and asks what it is:
- a flagged Q&A answer: answerable (type the right answer), unanswerable,
  false premise (type the correction), or no case -> evals/qa_flagged.jsonl,
  which the Q&A eval and the CI gate then include
- a sentence the Critic removed: was the Critic right? ->
  data/evals/judge_flagged.jsonl, which run_judge_eval.py then includes
- a reviewer edit: keep it as a review case? -> evals/review_labels.jsonl,
  run with run_review_eval.py --set evals/review_labels.jsonl
Each decision is saved as it's made: quit any time and pick up later.
"""

import argparse
import textwrap
from pathlib import Path

from langfuse import get_client

from arxiv_agent.evals.flywheel import (
    CANDIDATES,
    Candidate,
    judge_case,
    load_candidates,
    qa_item,
    review_case,
    save_candidates,
)
from arxiv_agent.evals.review_eval import SurveyCase

QA_FLAGGED = Path("evals/qa_flagged.jsonl")
JUDGE_FLAGGED = Path("data/evals/judge_flagged.jsonl")
REVIEW_LABELS = Path("evals/review_labels.jsonl")


def append(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(record.model_dump_json() + "\n")


def upsert_review_case(path: Path, case: SurveyCase) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    cases = [SurveyCase.model_validate_json(line) for line in lines if line.strip()]
    cases = [c for c in cases if c.id != case.id] + [case]
    path.write_text(
        "".join(c.model_dump_json() + "\n" for c in cases),
        encoding="utf-8",
        newline="\n",
    )


def ask(prompt: str, choices: str) -> str:
    while True:
        answer = input(f"{prompt} [{'/'.join(choices)}] ").strip().lower()[:1]
        if answer in choices:
            return answer


def show(candidate: Candidate, trace_url) -> None:
    wrap = textwrap.TextWrapper(width=100, initial_indent="  ", subsequent_indent="  ")
    print(f"\n[{candidate.kind}] {candidate.found}  {candidate.reason}")
    print(f"question: {candidate.question}")
    if candidate.kind == "qa":
        print(f"paper: {candidate.arxiv_id}v{candidate.version}\nanswer:")
        print(wrap.fill(candidate.answer or ""))
    elif candidate.kind == "critic":
        print("removed sentence:")
        print(wrap.fill(candidate.sentence or ""))
        for source, passage in zip(candidate.sources, candidate.passages, strict=True):
            print(f"cited passage ({source}):")
            print(wrap.fill(passage[:800]))
    else:
        print(f"paper: {candidate.arxiv_id} {candidate.title}")
    if candidate.trace_id:
        print(f"trace: {trace_url(candidate.trace_id)}")


def triage(candidate: Candidate, candidates: list[Candidate]) -> str | None:
    # Returns the new status, or None to skip for now.
    if candidate.kind == "qa":
        choice = ask(
            "answerable, unanswerable, false premise, no case, skip, quit?", "aufnsq"
        )
        if choice in "sq":
            return choice
        if choice == "n":
            return "rejected"
        item_type = {"a": "answerable", "u": "unanswerable", "f": "false_premise"}
        gold = [] if choice == "u" else [input("the right answer: ").strip()]
        append(QA_FLAGGED, qa_item(candidate, item_type[choice], gold))
        return "accepted"
    if candidate.kind == "critic":
        if not candidate.passages:
            print("(no cited passages stored: can't become a judge case)")
            return "rejected"
        choice = ask(
            "Critic right (not supported), wrong (supported), no case, skip, quit?",
            "rwnsq",
        )
        if choice in "sq":
            return choice
        if choice == "n":
            return "rejected"
        append(JUDGE_FLAGGED, judge_case(candidate, supported=choice == "w"))
        return "accepted"
    choice = ask("keep as a review case, no case, skip, quit?", "knsq")
    if choice in "sq":
        return choice
    if choice == "n":
        return "rejected"
    accepted = [
        c
        for c in candidates
        if c.kind == "label"
        and c.job_id == candidate.job_id
        and (c.status == "accepted" or c.id == candidate.id)
    ]
    upsert_review_case(REVIEW_LABELS, review_case(accepted))
    return "accepted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--list", action="store_true", help="list, don't ask")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    pending = [c for c in candidates if c.status == "pending"]
    print(f"{len(pending)} pending of {len(candidates)}")
    if args.list:
        for c in pending:
            print(f"  [{c.kind}] {c.found} {c.question[:60]} | {c.reason[:80]}")
        return
    langfuse = get_client()
    for candidate in pending:
        show(candidate, lambda trace_id: langfuse.get_trace_url(trace_id=trace_id))
        status = triage(candidate, candidates)
        if status == "q":
            break
        if status in ("accepted", "rejected"):
            candidate.status = status
            save_candidates(candidates, args.candidates)


if __name__ == "__main__":
    main()
