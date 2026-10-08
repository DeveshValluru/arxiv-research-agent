"""Run the Q&A eval set and print a report.

    uv run --env-file .env python scripts/run_eval.py
    uv run --env-file .env python scripts/run_eval.py --limit 5
    uv run --env-file .env python scripts/run_eval.py --model meta-llama/Llama-3.1-8B-Instruct --providers deepinfra
    uv run --env-file .env python scripts/run_eval.py --repeats 3 --concurrency 4

Every question goes through the real Answerer (traced in Langfuse, environment
"eval", one session per run), gets scored, and its scores are attached to its
trace. Results go to data/eval_runs/<run id>/: one line per question run, plus
a summary stamped with every version that can change a score. Latency is only
comparable between runs with the same --concurrency (default 1).
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from langfuse import get_client

from arxiv_agent.evals.judge import JUDGE_MODEL, JUDGE_PROMPT_VERSION, Judge
from arxiv_agent.evals.provenance import content_hash, git_version
from arxiv_agent.evals.qa_run import (
    MODEL,
    PROVIDERS,
    build_answerer,
    eval_sets,
    run_questions,
)
from arxiv_agent.evals.runner import ItemScore, load_items, summarize
from arxiv_agent.ingestion.chunker import CHUNKER_VERSION
from arxiv_agent.ingestion.embedder import DEFAULT_MODEL_ID
from arxiv_agent.qa.answerer import PROMPT_VERSION, REPAIR_PROMPT_VERSION
from arxiv_agent.storage.chunk_store import ChunkStore

RUNS_DIR = Path("data/eval_runs")
WORST_SHOWN = 8
ID_WIDTH = 24  # QASPER ids are 47 characters; the start is enough to find one


def fmt(value: float | None, pattern: str = "{:.2f}") -> str:
    return "  -  " if value is None else pattern.format(value)


def print_report(summary: dict, meta: dict, scores: list[ItemScore], langfuse) -> None:
    by_type = summary["correctness_by_type"]
    by_source = summary["correctness_by_source"]
    print(
        f"\nRun {meta['run_id']} | {meta['model']} via {meta['providers']} | "
        f"judge {meta['judge_model'].split('/')[-1]} | git {meta['git']}"
    )
    print(
        f"{summary['questions']} questions in {meta['duration_s']:.0f} s | "
        f"run failures {summary['run_failures']} | "
        f"judge failures {summary['judge_failures']}\n"
    )
    print(
        f"Correctness         {fmt(summary['correctness'])}   (answerable "
        f"{fmt(by_type['answerable'])} | unanswerable {fmt(by_type['unanswerable'])}"
        f" | false premise {fmt(by_type['false_premise'])})"
    )
    print(
        "  by source                "
        + " | ".join(f"{source} {fmt(value)}" for source, value in by_source.items())
    )
    print(
        f"Refusal accuracy    {fmt(summary['refusal_accuracy'])}"
        "   (unanswerable questions refused)"
    )
    print(
        f"False refusal rate  {fmt(summary['false_refusal_rate'])}"
        "   (answerable questions refused)"
    )
    print(
        f"Evidence recall@k   {fmt(summary['evidence_recall'])}   "
        f"({summary['evidence_unknown']} with unmatched evidence left out)"
    )
    print(f"Token F1            {fmt(summary['token_f1'])}   (answerable, cross-check)")
    print(f"Invalid answers     {fmt(summary['invalid_rate'])}")
    print(
        f"Support check       failed sentences in {fmt(summary['support_flagged_rate'])}"
        f" of answers; {summary['support_repaired']} repaired, "
        f"{summary['support_refusals']} turned into refusals"
    )
    print(f"Answer cost         ${summary['cost_usd']:.4f}")
    print(
        f"Latency             p50 {fmt(summary['latency_ms_p50'], '{:.0f}')} ms | "
        f"p95 {fmt(summary['latency_ms_p95'], '{:.0f}')} ms"
    )

    worst = sorted(
        (s for s in scores if s.correctness is None or s.correctness < 1),
        key=lambda s: -1 if s.correctness is None else s.correctness,
    )[:WORST_SHOWN]
    if worst:
        print("\nWorth a look:")
    for s in worst:
        reason = s.error or s.judge_reasoning or f"{s.type}, {s.status}"
        print(f"  {s.id[:ID_WIDTH]:{ID_WIDTH}} {fmt(s.correctness)}  {reason[:90]}")
        if s.trace_id:
            print(
                f"  {'':{ID_WIDTH}}        {langfuse.get_trace_url(trace_id=s.trace_id)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="+", type=Path, default=eval_sets())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--providers", default=PROVIDERS)
    parser.add_argument("-k", type=int, default=5, help="sources sent to the model")
    parser.add_argument(
        "--retriever", choices=["dense", "hybrid", "rerank"], default="rerank"
    )
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument(
        "--no-support-check",
        action="store_true",
        help="skip the claim-support check (to measure what it changes)",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="remove failing sentences without asking for a rewrite first",
    )
    parser.add_argument("--repeats", type=int, default=1, help="runs per question")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
    langfuse = get_client()
    started_at = datetime.now(UTC)
    model_name = args.model.split("/")[-1].lower()
    run_id = f"{started_at:%Y%m%d-%H%M%S}-{model_name}-{args.retriever}"

    items = load_items(args.sets)[: args.limit]
    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    answerer = build_answerer(
        store,
        langfuse,
        model=args.model,
        providers=args.providers.split(","),
        k=args.k,
        retriever=args.retriever,
        support=not args.no_support_check,
        repair=not args.no_repair,
    )
    done = 0

    def progress(score: ItemScore) -> None:
        nonlocal done
        done += 1
        print(
            f"[{done:3}/{len(items) * args.repeats}] {score.id[:ID_WIDTH]:{ID_WIDTH}} "
            f"{score.status or 'FAILED':9} {fmt(score.correctness)}",
            flush=True,
        )

    start = time.perf_counter()
    scores = run_questions(
        items,
        answerer,
        Judge(langfuse=langfuse),
        store,
        run_id=run_id,
        langfuse=langfuse,
        repeats=args.repeats,
        concurrency=args.concurrency,
        progress=progress,
    )
    store.close()
    langfuse.flush()

    summary = summarize(scores)
    meta = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "duration_s": time.perf_counter() - start,
        "git": git_version(),
        "model": args.model,
        "providers": args.providers,
        "k": args.k,
        "prompt_version": PROMPT_VERSION,
        "embedder": DEFAULT_MODEL_ID,
        "retriever": answerer.retriever_name,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "support_check": not args.no_support_check,
        "support_repair": not (args.no_support_check or args.no_repair),
        "repair_prompt_version": REPAIR_PROMPT_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "judge_model": JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "eval_sets": {path.as_posix(): content_hash(path) for path in args.sets},
    }
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(score.model_dump_json() + "\n" for score in scores)
    (run_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print_report(summary, meta, scores, langfuse)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
