"""Judge the judge: does the Critic catch sentences we broke on purpose?

    uv run --env-file .env python scripts/build_judge_eval.py    (once)
    uv run --env-file .env python scripts/run_judge_eval.py
    uv run --env-file .env python scripts/run_judge_eval.py --judges production --limit 20

Every case (see arxiv_agent/evals/judge_eval.py) goes to each judge with the
production Critic prompt:
- production: the review's judge alone (Llama-3.3-70B, a different model
  family from the writer)
- critic: what the Critic does, the number check in code first, then that judge
- same-family: Qwen3-32B, the writer's own model, for comparison
Results go to data/eval_runs/<run id>/. 200 cases per judge: a few minutes each.
"""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from langfuse import get_client, propagate_attributes

from arxiv_agent.evals.judge import JUDGE_MODEL
from arxiv_agent.evals.judge_eval import (
    KINDS,
    JudgedCase,
    SupportCase,
    judge_case,
    summarize,
)
from arxiv_agent.evals.provenance import git_version
from arxiv_agent.llm import ChatModel
from arxiv_agent.review.critic import CRITIC_PROMPT, CRITIC_PROMPT_VERSION
from arxiv_agent.review.setup import ReviewSettings

CASES = Path("data/evals/judge_support.jsonl")
# Real Critic removals a person labeled (scripts/triage_flags.py, 7.3).
FLAGGED = Path("data/evals/judge_flagged.jsonl")
RUNS_DIR = Path("data/eval_runs")
SHOWN_MISSES = 6


def judges(
    settings: ReviewSettings, langfuse
) -> dict[str, tuple[ChatModel, str, bool]]:
    # name -> (model, prompt, run the Critic's number check first)
    production = ChatModel(
        JUDGE_MODEL, settings.judge_providers, langfuse=langfuse, timeout=30
    )
    return {
        "production": (production, CRITIC_PROMPT, False),  # the judge alone
        "critic": (production, CRITIC_PROMPT, True),  # what the Critic does
        # Qwen3 thinks before answering unless told not to; the JSON then
        # wouldn't fit in the judge's 150 tokens.
        "same-family": (
            ChatModel(
                settings.model, settings.providers, langfuse=langfuse, timeout=60
            ),
            CRITIC_PROMPT + " /no_think",
            False,
        ),
    }


def fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.2f}"


def print_report(
    results: dict[str, list[JudgedCase]], cases: dict[str, SupportCase]
) -> None:
    header = "".join(f"{kind[:13]:>15}" for kind in KINDS)
    print(f"\n{'judge':<13}{'caught':>8}{'false alarms':>14}{header}{'errors':>8}")
    for name, judged in results.items():
        s = summarize(judged)
        by_kind = "".join(f"{fmt(s['by_kind'][k]['correct']):>15}" for k in KINDS)
        print(
            f"{name:<13}{fmt(s['caught']):>8}{fmt(s['false_alarms']):>14}{by_kind}{s['errors']:>8}"
        )
    print(
        "\ncaught: broken sentences judged not supported | false alarms: true "
        "sentences judged not supported\nper kind: share judged right"
    )
    for name, judged in results.items():
        misses = [j for j in judged if not j.supported and j.verdict == "supported"]
        if misses:
            print(f"\n{name}: broken sentences it passed ({len(misses)}), e.g.")
        for miss in misses[:SHOWN_MISSES]:
            print(f"  [{miss.kind}] {cases[miss.id].sentence[:130]}")
            print(f"      judge: {miss.reason[:120]}")


async def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
    langfuse = get_client()
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}-judge"
    lines = [
        line
        for path in args.cases
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    cases = [SupportCase.model_validate_json(line) for line in lines if line.strip()]
    cases = cases[: args.limit] if args.limit else cases
    available = judges(ReviewSettings(), langfuse)

    results: dict[str, list[JudgedCase]] = {}
    limit = asyncio.Semaphore(args.concurrency)
    for name in args.judges:
        judge, prompt, number_check = available[name]
        with propagate_attributes(session_id=run_id, tags=[f"judge:{name}"]):
            results[name] = await asyncio.gather(
                *(
                    judge_case(judge, case, limit, prompt, number_check)
                    for case in cases
                )
            )
        print(f"{name}: {len(results[name])} cases judged", flush=True)
    langfuse.flush()

    out = args.out / run_id
    out.mkdir(parents=True)
    for name, judged in results.items():
        (out / f"{name}.jsonl").write_text(
            "".join(j.model_dump_json() + "\n" for j in judged), encoding="utf-8"
        )
    summary = {
        "meta": {
            "run_id": run_id,
            "git": git_version(),
            "cases": len(cases),
            "critic_prompt_version": CRITIC_PROMPT_VERSION,
        },
        "judges": {name: summarize(judged) for name, judged in results.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_report(results, {case.id: case for case in cases})
    print(f"\nresults: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cases",
        type=Path,
        nargs="+",
        default=[CASES, *([FLAGGED] if FLAGGED.exists() else [])],
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["production", "critic", "same-family"],
        choices=["production", "critic", "same-family"],
    )
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    asyncio.run(main(parser.parse_args()))
