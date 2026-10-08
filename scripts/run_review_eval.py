"""Score literature reviews against survey papers' bibliographies.

    uv run --env-file .env python scripts/run_review_eval.py
    uv run --env-file .env python scripts/run_review_eval.py --cases rag tool-learning
    uv run --env-file .env python scripts/run_review_eval.py --set evals/review_labels.jsonl

The second set is built from reviewers' edits (scripts/triage_flags.py, 7.3):
the gold is what a reviewer had to add, and keeping a paper they removed is
reported.

Each case runs a review of a survey's question in quick mode (no pause: this
measures the system alone), as of the survey's first version date, and scores
it stage by stage against the survey's bibliography (see
arxiv_agent/evals/review_eval.py). Traced in Langfuse (environment "eval", one
session per run), with the scores attached to each review's trace. Results go
to data/eval_runs/<run id>/: a line per case, each review's full result, and a
summary stamped with every version that can change a score.

A full run is 5 reviews: about 15 to 20 minutes.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from langfuse import Langfuse, get_client

from arxiv_agent.evals.judge import JUDGE_MODEL
from arxiv_agent.evals.provenance import content_hash, git_version
from arxiv_agent.evals.review_eval import (
    METRICS,
    ReviewScore,
    load_cases,
    score_review,
    summarize,
)
from arxiv_agent.ingestion.chunker import CHUNKER_VERSION
from arxiv_agent.ingestion.embedder import DEFAULT_MODEL_ID
from arxiv_agent.review.checkpoints import memory_checkpointer
from arxiv_agent.review.events import describe_event, result_of
from arxiv_agent.review.graph import run_review
from arxiv_agent.review.setup import ReviewSettings, open_review_graph
from arxiv_agent.review.state import Budget

EVAL_SET = Path("evals/review_surveys.jsonl")
RUNS_DIR = Path("data/eval_runs")

logger = logging.getLogger(__name__)


def attach_scores(langfuse: Langfuse, score: ReviewScore, run_id: str) -> None:
    if score.trace_id is None:
        return
    for metric in METRICS:
        value = getattr(score, metric)
        if value is not None:
            langfuse.create_score(
                name=metric,
                value=value,
                trace_id=score.trace_id,
                metadata={"run_id": run_id, "case": score.id},
            )


def fmt(value: float | None, pattern: str = "{:.2f}") -> str:
    return "  -  " if value is None else pattern.format(value)


def print_report(scores: list[ReviewScore], summary: dict, meta: dict) -> None:
    print(
        f"\nRun {meta['run_id']} | {meta['model'].split('/')[-1]}, judge "
        f"{meta['judge_model'].split('/')[-1]} | git {meta['git']} | "
        f"{summary['cases']} reviews in {summary['seconds'] / 60:.0f} min, "
        f"{summary['failures']} failed\n"
    )
    print(
        f"{'case':<14}{'gold':>5}{'cand':>6}  search  +snowball   kept      lift  "
        f"cited     support  calls"
    )
    for s in scores:
        if s.error:
            print(f"{s.id:<14}{s.gold:>5}  FAILED: {s.error[:80]}")
            continue
        print(
            f"{s.id:<14}{s.gold:>5}{s.candidates:>6}  {fmt(s.search_recall)}    "
            f"{fmt(s.candidate_recall)}      {s.kept_gold}/{s.kept} {fmt(s.kept_precision)}  "
            f"{fmt(s.screener_lift, '{:.1f}')}   {s.cited_gold}/{s.cited} {fmt(s.cited_precision)}  "
            f"{fmt(s.support_rate)}   {s.llm_calls:>4}"
            + (f"  STOPPED: {s.stopped}" if s.stopped else "")
        )
        if s.excluded_kept:  # cases from review labels (7.3)
            print(f"{'':<14}kept again what a reviewer removed: {s.excluded_kept}")
    print(
        f"{'mean':<25}  {fmt(summary['search_recall'])}    "
        f"{fmt(summary['candidate_recall'])}           {fmt(summary['kept_precision'])}  "
        f"{fmt(summary['screener_lift'], '{:.1f}')}       {fmt(summary['cited_precision'])}  "
        f"{fmt(summary['support_rate'])}"
    )
    print(
        "\nsearch / +snowball: share of the survey's arXiv-cited papers found "
        "(by search; with snowballing)\nkept, cited: expert-cited / total, and "
        "that share | lift: kept share / candidate share (1 = chance)"
    )


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
    langfuse = get_client()
    settings = ReviewSettings()
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}-review-{settings.model.split('/')[-1].lower()}"
    cases = [c for c in load_cases(args.set) if not args.cases or c.id in args.cases]
    out = args.out / run_id
    (out / "results").mkdir(parents=True)

    scores: list[ReviewScore] = []
    async with open_review_graph(settings, langfuse, memory_checkpointer()) as graph:
        for n, case in enumerate(cases, start=1):
            print(
                f"\n[{n}/{len(cases)}] {case.id}, as of {case.cutoff}, "
                f"{len(case.gold)} gold papers: {case.question}",
                flush=True,
            )
            start = time.monotonic()

            def show(kind: str, data: dict, start: float = start) -> None:
                if kind == "step":
                    elapsed = time.monotonic() - start
                    print(
                        f"  [{elapsed:4.0f}s] {describe_event(kind, data)[:110]}",
                        flush=True,
                    )

            try:
                run = await run_review(
                    case.question,
                    graph,
                    thread_id=f"{run_id}-{case.id}",
                    budget=Budget(),
                    published_before=case.cutoff,
                    on_event=show,
                    session_id=run_id,
                    langfuse=langfuse,
                )
            except Exception as exc:
                # One failed review mustn't end the run: record it, move on.
                logger.exception("review for %s failed", case.id)
                score = ReviewScore(
                    id=case.id,
                    gold=len(case.gold),
                    seconds=round(time.monotonic() - start, 1),
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
            else:
                score = score_review(
                    case, run.state, time.monotonic() - start, run.trace_id
                )
                attach_scores(langfuse, score, run_id)
                result = json.dumps(result_of(run.state), indent=1, ensure_ascii=False)
                (out / "results" / f"{case.id}.json").write_text(
                    result, encoding="utf-8"
                )
            scores.append(score)
            with (out / "reviews.jsonl").open("a", encoding="utf-8") as lines:
                lines.write(score.model_dump_json() + "\n")
    langfuse.flush()

    summary = summarize(scores)
    meta = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "git": git_version(),
        "eval_set": {args.set.as_posix(): content_hash(args.set)},
        "model": settings.model,
        "providers": settings.providers,
        "judge_model": JUDGE_MODEL,
        "judge_providers": settings.judge_providers,
        "keep": settings.keep,
        "max_revisions": settings.max_revisions,
        "budget": Budget().model_dump(),
        "embedder": DEFAULT_MODEL_ID,
        "chunker_version": CHUNKER_VERSION,
    }
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8"
    )
    print_report(scores, summary, meta)
    print(f"\nresults: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", type=Path, default=EVAL_SET)
    parser.add_argument("--cases", nargs="+", help="case ids, e.g. rag tool-learning")
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    asyncio.run(main(parser.parse_args()))
