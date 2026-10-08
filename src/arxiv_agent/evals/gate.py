"""The CI eval gate: is a change worse than the eval's own noise?

One Q&A eval run is noisy. Identical configs scored 2 points apart in 6.3b,
and per question a grader flip or a different generation moves a score from
1 to 0. A fixed threshold either fails good changes or misses real
regressions, so the gate is paired and noise-aware:
- the baseline is k runs of every question on main: each question's mean
  score, and the run-to-run variance of a question's score, pooled
- a candidate runs the same questions r times. D is the mean over questions
  of (candidate mean - baseline mean): paired, so a hard question isn't a
  regression just for being hard
- D's standard error comes from that variance: SE = sqrt(v (1/r + 1/k) / n)
- fail if D < -max(floor, z * SE): a drop larger than noise explains, and at
  least the smallest drop worth blocking a change for
Runs the system or the judge couldn't finish (provider outages) make the
result inconclusive, and inconclusive fails: a gate that passes when it
couldn't measure isn't a gate. The baseline file holds scores, never answers.
"""

import hashlib
import math
from collections import defaultdict
from statistics import mean, variance
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.evals.runner import EvalItem, ItemScore

Z = 2.0  # one-sided: about 2% of pure-noise runs fail
# The smallest drop worth failing a change for, whatever the noise. Calibrated
# (7.2): unchanged runs moved at most 1.5 SE; 3 sources instead of 5 lost 5
# points on the full suite (2.7 SE), and a 5-point floor let it pass.
FLOOR = 0.03
MAX_UNSCORED = 0.1  # more unscored runs than this: inconclusive
# A baseline graded differently can't be compared with: these must match.
JUDGE_KEYS = ("judge_model", "judge_prompt_version")


class ItemStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: int
    mean: float
    var: float  # sample variance across runs; 0 with one run


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: dict
    pooled_var: float  # run-to-run variance of one question's score
    items: dict[str, ItemStats]  # correctness per question id


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    baseline: float
    candidate: float
    trace_id: str | None  # a candidate run, to see what it answered


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "fail", "inconclusive"]
    questions: int  # compared: scored in the candidate and in the baseline
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    delta: float | None = None
    se: float | None = None
    tolerance: float | None = None
    worse: list[Change] = []
    better: list[Change] = []
    new: list[str] = []  # no baseline yet: reported, not gated
    problems: list[str] = []


def stratified_subset(items: list[EvalItem], size: int) -> list[EvalItem]:
    # The PR suite: each (source, type) group in proportion, at least one
    # each, picked by a hash of the id, so the subset is the same every run
    # and a question added later doesn't reshuffle the rest.
    groups: dict[tuple[str, str], list[EvalItem]] = defaultdict(list)
    for item in items:
        groups[item.source, item.type].append(item)
    for group in groups.values():
        group.sort(key=lambda i: hashlib.sha1(i.id.encode()).hexdigest())
    if size >= len(items):
        return list(items)
    quotas = {key: size * len(group) / len(items) for key, group in groups.items()}
    taken = {key: max(1, math.floor(q)) for key, q in quotas.items()}
    by_remainder = sorted(quotas, key=lambda key: quotas[key] - math.floor(quotas[key]))
    while sum(taken.values()) < size and by_remainder:
        key = by_remainder.pop()
        if taken[key] < len(groups[key]):
            taken[key] += 1
    chosen = {i.id for key, group in groups.items() for i in group[: taken[key]]}
    return [item for item in items if item.id in chosen]


def item_stats(scores: list[ItemScore]) -> dict[str, ItemStats]:
    runs: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        if score.correctness is not None:
            runs[score.id].append(score.correctness)
    return {
        item_id: ItemStats(
            runs=len(values),
            mean=mean(values),
            var=variance(values) if len(values) > 1 else 0.0,
        )
        for item_id, values in runs.items()
    }


def pooled_variance(stats: dict[str, ItemStats]) -> float:
    # Each question's variance weighted by its degrees of freedom.
    dof = sum(s.runs - 1 for s in stats.values() if s.runs > 1)
    if not dof:
        return 0.0
    return sum(s.var * (s.runs - 1) for s in stats.values() if s.runs > 1) / dof


def make_baseline(scores: list[ItemScore], meta: dict) -> Baseline:
    stats = item_stats(scores)
    return Baseline(meta=meta, pooled_var=pooled_variance(stats), items=stats)


def _unscored(scores: list[ItemScore]) -> float:
    return sum(s.correctness is None for s in scores) / len(scores) if scores else 1.0


def compare(
    baseline: Baseline,
    scores: list[ItemScore],
    meta: dict,
    *,
    floor: float = FLOOR,
    z: float = Z,
) -> GateResult:
    problems = [
        f"{key} differs from the baseline's ({baseline.meta.get(key)} vs "
        f"{meta.get(key)}): scores aren't comparable; update the baseline"
        for key in JUDGE_KEYS
        if baseline.meta.get(key) != meta.get(key)
    ]
    if problems:
        return GateResult(status="fail", questions=0, problems=problems)
    unscored = _unscored(scores)
    if unscored > MAX_UNSCORED:
        return GateResult(
            status="inconclusive",
            questions=0,
            problems=[
                (
                    f"{unscored:.0%} of runs got no score (provider or judge "
                    "failures): rerun the gate"
                )
            ],
        )

    candidate = item_stats(scores)
    shared = sorted(set(candidate) & set(baseline.items))
    if not shared:
        return GateResult(
            status="inconclusive",
            questions=0,
            problems=["no question has both a baseline and a candidate score"],
        )
    repeats = mean(candidate[i].runs for i in shared)
    k = mean(baseline.items[i].runs for i in shared)
    n = len(shared)
    delta = mean(candidate[i].mean - baseline.items[i].mean for i in shared)
    se = math.sqrt(baseline.pooled_var * (1 / repeats + 1 / k) / n)
    tolerance = max(floor, z * se)

    traces: dict[str, str | None] = {}
    for score in scores:  # the lowest-scoring run of each question
        if score.correctness is not None and (
            score.id not in traces or score.correctness < candidate[score.id].mean
        ):
            traces[score.id] = score.trace_id

    def change(i: str) -> Change:
        return Change(
            id=i,
            baseline=baseline.items[i].mean,
            candidate=candidate[i].mean,
            trace_id=traces.get(i),
        )

    gaps = {i: candidate[i].mean - baseline.items[i].mean for i in shared}
    return GateResult(
        status="fail" if delta < -tolerance else "pass",
        questions=n,
        baseline_mean=mean(baseline.items[i].mean for i in shared),
        candidate_mean=mean(candidate[i].mean for i in shared),
        delta=delta,
        se=se,
        tolerance=tolerance,
        # Moves of half a point or more: worth a look either way.
        worse=[change(i) for i in sorted(shared, key=gaps.get) if gaps[i] <= -0.5],
        better=[change(i) for i in sorted(shared, key=gaps.get) if gaps[i] >= 0.5],
        new=sorted(set(candidate) - set(baseline.items)),
    )


def render_markdown(
    result: GateResult,
    meta: dict,
    trace_url=lambda trace_id: None,
    plain: bool = False,
) -> str:
    # For the CI job summary. plain: no icons or arrows, for a Windows console
    # whose output is redirected (cp1252 can't encode them).
    icon = {"pass": "✅ ", "fail": "❌ ", "inconclusive": "⚠️ "}[result.status]
    icon, arrow = ("", "->") if plain else (icon, "→")
    lines = [f"## {icon}Q&A eval gate: {result.status}", ""]
    if result.delta is not None:
        lines += [
            (
                f"{result.questions} questions x {meta.get('repeats')} runs "
                f"({meta.get('suite')} suite) against the baseline on main."
            ),
            "",
            "| | correctness |",
            "|---|---|",
            f"| baseline | {result.baseline_mean:.3f} |",
            f"| this change | {result.candidate_mean:.3f} |",
            f"| difference | {result.delta:+.3f} (noise SE {result.se:.3f}) |",
            f"| fails below | {-result.tolerance:+.3f} |",
            "",
        ]
    lines += [f"- {problem}" for problem in result.problems]
    for title, changes in (("Worse", result.worse), ("Better", result.better)):
        if changes:
            lines += ["", f"**{title}** (moved half a point or more):", ""]
        for c in changes:
            url = trace_url(c.trace_id) if c.trace_id else None
            where = f" ([trace]({url}))" if url else ""
            lines.append(
                f"- `{c.id}`: {c.baseline:.2f} {arrow} {c.candidate:.2f}{where}"
            )
    if result.new:
        lines += [
            "",
            f"New questions without a baseline (not gated): {len(result.new)}",
        ]
    lines += [
        "",
        (
            f"<sub>model {meta.get('model')} · prompt v{meta.get('prompt_version')} · "
            f"judge {str(meta.get('judge_model')).split('/')[-1]} "
            f"v{meta.get('judge_prompt_version')} · git {meta.get('git')}</sub>"
        ),
    ]
    return "\n".join(lines) + "\n"
