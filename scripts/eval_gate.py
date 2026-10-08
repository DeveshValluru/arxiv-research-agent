"""The Q&A eval gate: fail when a change makes answers worse than noise explains.

    uv run --env-file .env python scripts/eval_gate.py --suite pr
    uv run --env-file .env python scripts/eval_gate.py --suite full
    uv run --env-file .env python scripts/eval_gate.py --update-baseline

pr: a stratified 20-question subset, each asked twice (CI on pull requests).
full: every question, twice (CI on main). Both compare with the committed
baseline (evals/baselines/qa.json) as arxiv_agent/evals/gate.py describes,
print a report, write it to the CI job summary when there is one, and exit 1
on fail or inconclusive.

--update-baseline asks every question three times and writes the baseline
instead: run it on main, and commit the file with the change that moved it
(a new prompt, model or judge). Config flags (--k, --model, --no-repair, ...)
try a change without editing code, e.g. to check the gate catches a bad one.
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from langfuse import get_client

from arxiv_agent.evals.gate import (
    Baseline,
    compare,
    make_baseline,
    render_markdown,
    stratified_subset,
)
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
from arxiv_agent.qa.answerer import PROMPT_VERSION, REPAIR_PROMPT_VERSION
from arxiv_agent.review.critic import CRITIC_PROMPT_VERSION
from arxiv_agent.storage.chunk_store import ChunkStore

BASELINE = Path("evals/baselines/qa.json")
RUNS_DIR = Path("data/eval_runs")
PR_QUESTIONS = 20
REPEATS = {"pr": 2, "full": 2, "baseline": 3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=["pr", "full"], default="pr")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--repeats", type=int, help="runs per question")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--summary",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="append the markdown report here (CI sets GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--providers", default=PROVIDERS)
    parser.add_argument("-k", type=int, default=5, help="sources sent to the model")
    parser.add_argument(
        "--retriever", choices=["dense", "hybrid", "rerank"], default="rerank"
    )
    parser.add_argument("--no-support-check", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    suite = "baseline" if args.update_baseline else args.suite
    repeats = args.repeats or REPEATS[suite]
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
    langfuse = get_client()
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-gate-{suite}"

    sets = eval_sets()
    items = load_items(sets)
    if suite == "pr":
        items = stratified_subset(items, PR_QUESTIONS)
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
    meta = {
        "run_id": run_id,
        "suite": suite,
        "repeats": repeats,
        "git": git_version(),
        "model": args.model,
        "providers": args.providers,
        "k": args.k,
        "retriever": args.retriever,
        "support_check": not args.no_support_check,
        "support_repair": not (args.no_support_check or args.no_repair),
        "prompt_version": PROMPT_VERSION,
        "repair_prompt_version": REPAIR_PROMPT_VERSION,
        "critic_prompt_version": CRITIC_PROMPT_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "judge_model": JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "eval_sets": {path.as_posix(): content_hash(path) for path in sets},
    }

    def progress(score: ItemScore) -> None:
        mark = "-" if score.correctness is None else f"{score.correctness:.1f}"
        print(f"  {score.id[:40]:40} run {score.repeat + 1}  {mark}", flush=True)

    print(f"{suite}: {len(items)} questions x {repeats} runs", flush=True)
    start = time.perf_counter()
    scores = run_questions(
        items,
        answerer,
        Judge(langfuse=langfuse),
        store,
        run_id=run_id,
        langfuse=langfuse,
        repeats=repeats,
        concurrency=args.concurrency,
        progress=progress,
    )
    store.close()
    langfuse.flush()
    meta["duration_s"] = round(time.perf_counter() - start)

    out = args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text(
        "".join(score.model_dump_json() + "\n" for score in scores), encoding="utf-8"
    )
    summary = summarize(scores)

    if args.update_baseline:
        baseline = make_baseline(scores, meta | {"summary": summary})
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            baseline.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"\nbaseline: {len(baseline.items)} questions x {repeats} runs | "
            f"correctness {summary['correctness']:.3f} | run-to-run variance per "
            f"question {baseline.pooled_var:.4f}\nwrote {args.baseline}"
        )
        return 0

    baseline = Baseline.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    result = compare(baseline, scores, meta)
    (out / "gate.json").write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "gate": result.model_dump()}, indent=2
        ),
        encoding="utf-8",
    )

    def trace_url(trace_id: str) -> str:
        return langfuse.get_trace_url(trace_id=trace_id)

    print("\n" + render_markdown(result, meta, trace_url, plain=True))
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as f:
            f.write(render_markdown(result, meta, trace_url))
    print(f"results: {out}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
