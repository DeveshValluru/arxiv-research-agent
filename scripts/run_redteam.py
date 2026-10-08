"""Red-team eval: can instructions planted in a paper steer a review?

    uv run --env-file .env python scripts/run_redteam.py

Runs every attack in evals/redteam.json through the real models (see
arxiv_agent/evals/redteam.py): Screener attacks (does an off-topic paper with
an injection get kept?) and review attacks (does injected content reach the
shipped review, and which layer stops it?). Traced in Langfuse (environment
"eval", one session per run). Results go to data/eval_runs/<run id>/.

--no-content-guard turns the content guard off, to measure what it stops.
About 20 screening calls and 5 short reviews: 10 to 15 minutes.

--generated adds the injection eval's generated attacks whose goal is to get
a paper kept or scored high (evals/injection_attacks.jsonl, 6.4) as Screener
attacks: written without seeing the content guard's patterns.
--screener-only skips the review attacks.
"""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from langfuse import get_client, propagate_attributes

from arxiv_agent.evals.injection_eval import Attack
from arxiv_agent.evals.judge import JUDGE_MODEL
from arxiv_agent.evals.provenance import content_hash, git_version
from arxiv_agent.evals.redteam import (
    Injection,
    ReviewAttack,
    ScreenerAttack,
    load_red_team,
    review_attack,
    review_library,
    screener_attack,
)
from arxiv_agent.guardrails.output import OutputGuard
from arxiv_agent.llm import ChatModel
from arxiv_agent.review.checkpoints import memory_checkpointer
from arxiv_agent.review.critic import Critic
from arxiv_agent.review.graph import REVIEW_PROMPTS, build_review_graph
from arxiv_agent.review.human import HumanReview
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.reader import Reader
from arxiv_agent.review.setup import ReviewSettings
from arxiv_agent.review.writer import Synthesizer

RED_TEAM = Path("evals/redteam.json")
GENERATED = Path("evals/injection_attacks.jsonl")
SCREENING_GOALS = {"keep", "score"}
RUNS_DIR = Path("data/eval_runs")


def yes(flag: bool) -> str:
    return "yes" if flag else " - "


def print_report(
    screener: list[ScreenerAttack], review: list[ReviewAttack], targets: list[str]
) -> None:
    print("\nScreener attacks: an off-topic paper whose abstract carries an injection")
    print(
        "(score out of 10; * = kept, i.e. the attack worked; 'clean' is the control)\n"
    )
    print(f"{'injection':<26}" + "".join(f"{t:>12}" for t in targets) + "   kept")
    for injection in dict.fromkeys(r.injection for r in screener):
        row = [r for r in screener if r.injection == injection]
        cells = "".join(f"{r.score:>11}{'*' if r.kept else ' '}" for r in row)
        print(f"{injection:<26}{cells}   {sum(r.kept for r in row)}/{len(row)}")

    print("\nReview attacks: a kept paper's text carries an injection")
    print("(where the injected content got to; shipped = the attack worked)\n")
    print(
        f"{'injection':<16}{'claims':>8}{'draft':>8}{'shipped':>9}{'removed':>9}{'blocked':>9}"
    )
    for r in review:
        print(
            f"{r.injection:<16}{yes(r.in_claims):>8}{yes(r.in_draft):>8}"
            f"{yes(r.shipped):>9}{r.removed:>9}{r.blocked:>9}"
        )
    attacks = [r for r in screener if r.injection != "clean"]
    print(
        f"\nAttack success: Screener {sum(r.kept for r in attacks)}/{len(attacks)}, "
        f"review {sum(r.shipped for r in review)}/{len(review)}"
    )


async def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
    langfuse = get_client()
    settings = ReviewSettings()
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}-redteam"
    data = load_red_team(args.set)
    writer = ChatModel(
        settings.model, settings.providers, langfuse=langfuse, timeout=120
    )
    judge = ChatModel(
        JUDGE_MODEL, settings.judge_providers, langfuse=langfuse, timeout=30
    )
    guard_content = not args.no_content_guard
    nodes = ReviewNodes(  # screening only
        writer, toolbox=None, guard_content=guard_content, langfuse=langfuse
    )

    injections = list(data.screener_injections)
    if args.generated:
        lines = GENERATED.read_text(encoding="utf-8").splitlines()
        attacks = [Attack.model_validate_json(line) for line in lines if line.strip()]
        injections += [
            Injection(id=a.id, text=a.text)
            for a in attacks
            if a.goal in SCREENING_GOALS
        ]
    limit = asyncio.Semaphore(args.concurrency)

    async def attack(target, injection) -> ScreenerAttack:
        async with limit:
            with (
                langfuse.start_as_current_observation(
                    as_type="agent",
                    name="red-team-screener",
                    input={"target": target.arxiv_id, "injection": injection.id},
                ) as span,
                propagate_attributes(session_id=run_id),
            ):
                result = await screener_attack(nodes, data, target, injection)
                span.update(output=result.model_dump())
        print(
            f"screener  {target.arxiv_id}  {injection.id:<26} score {result.score:>2}  kept {result.kept}",
            flush=True,
        )
        return result

    # Grouped by injection, so the report reads one attack per row.
    screener: list[ScreenerAttack] = await asyncio.gather(
        *(
            attack(target, injection)
            for injection in injections
            for target in data.off_topic
        )
    )

    review: list[ReviewAttack] = []
    leak_guard = OutputGuard(REVIEW_PROMPTS)
    for injection in [] if args.screener_only else data.review_injections:
        graph = build_review_graph(
            nodes,
            Reader(
                writer,
                review_library(data, injection),
                guard_content=guard_content,
                langfuse=langfuse,
            ),
            Synthesizer(writer, langfuse=langfuse),
            Critic(judge, langfuse=langfuse),
            HumanReview(toolbox=None, langfuse=langfuse),
            checkpointer=memory_checkpointer(),
        )
        with (
            langfuse.start_as_current_observation(
                as_type="agent",
                name="red-team-review",
                input={"injection": injection.id},
            ) as span,
            propagate_attributes(session_id=run_id),
        ):
            result = await review_attack(
                graph, data, injection, leak_guard, thread_id=f"{run_id}-{injection.id}"
            )
            span.update(output=result.model_dump())
        review.append(result)
        print(f"review    {injection.id:<16} shipped {result.shipped}", flush=True)
    langfuse.flush()

    out = args.out / run_id
    out.mkdir(parents=True)
    report = {
        "meta": {
            "run_id": run_id,
            "git": git_version(),
            "red_team": {args.set.as_posix(): content_hash(args.set)},
            "model": settings.model,
            "judge_model": JUDGE_MODEL,
            "content_guard": guard_content,
            "generated": args.generated,
        },
        "screener": [r.model_dump() for r in screener],
        "review": [r.model_dump() for r in review],
    }
    (out / "redteam.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nContent guard: {'on' if guard_content else 'OFF'}")
    print_report(screener, review, [p.arxiv_id for p in data.off_topic])
    print(f"\nresults: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", type=Path, default=RED_TEAM)
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    parser.add_argument("--no-content-guard", action="store_true")
    parser.add_argument("--generated", action="store_true")
    parser.add_argument("--screener-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
