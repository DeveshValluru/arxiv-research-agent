import math
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.evals.judge import SCORES, Judge, JudgeError
from arxiv_agent.evals.qa_scoring import (
    best_f1,
    evidence_hit,
    gold_chunk_ids,
    refusal_correct,
)
from arxiv_agent.ingestion.models import Chunk
from arxiv_agent.qa.answerer import QAResult
from arxiv_agent.qa.support import FAILING

ItemType = Literal["answerable", "unanswerable", "false_premise"]
TYPES: tuple[ItemType, ...] = ("answerable", "unanswerable", "false_premise")
FALSE_PREMISE_REFUSAL_SCORE = 0.5  # safe, but it doesn't correct the premise


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    arxiv_id: str
    version: int
    question: str
    type: ItemType
    answer_type: str | None
    gold_answers: list[str]
    evidence: list[str]
    evidence_matched: bool


class ItemScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    type: ItemType
    repeat: int = 0  # which run of this question (the gate asks each one r times)
    status: str | None = None  # None: no answer at all (the run failed)
    answer: str | None = None
    correctness: float | None = None  # 1, 0.5 or 0; None: not graded
    graded_by: Literal["rule", "judge"] | None = None
    judge_reasoning: str | None = None
    f1: float | None = None
    refusal_ok: bool | None = None
    evidence_hit: bool | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    flagged_sentences: int = 0  # failed by the claim-support check
    repaired: bool = False  # the rewrite of the failing sentences was used
    support_refused: bool = False  # the check left nothing, so the answer refused
    trace_id: str | None = None
    error: str | None = None


def load_items(paths: Iterable[Path]) -> list[EvalItem]:
    return [
        EvalItem.model_validate_json(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def grade(
    item: EvalItem, status: str, answer: str, judge: Judge
) -> tuple[float, str, str | None]:
    # Rules where the answer is certain, the judge only where it takes reading:
    # cheaper, and no judge bias on the cases a string comparison settles.
    if item.type == "unanswerable":
        return (1.0 if status == "refused" else 0.0), "rule", None
    if status == "refused":
        score = FALSE_PREMISE_REFUSAL_SCORE if item.type == "false_premise" else 0.0
        return score, "rule", None
    verdict = judge.grade(
        item.question,
        item.gold_answers,
        answer,
        false_premise=item.type == "false_premise",
    )
    return SCORES[verdict.verdict], "judge", verdict.reasoning


def score_item(
    item: EvalItem, result: QAResult, chunks: list[Chunk], judge: Judge
) -> ItemScore:
    status, answer = result.answer.status, result.answer.text
    flagged = sum(s.verdict in FAILING for s in result.support)
    score = graded_by = reasoning = error = None
    try:
        score, graded_by, reasoning = grade(item, status, answer, judge)
    except JudgeError as exc:
        error = f"judge: {exc}"  # counted as a judge failure, never as "incorrect"

    return ItemScore(
        id=item.id,
        source=item.source,
        type=item.type,
        status=status,
        answer=answer,
        correctness=score,
        graded_by=graded_by,
        judge_reasoning=reasoning,
        f1=best_f1(answer, item.gold_answers) if item.type == "answerable" else None,
        refusal_ok=refusal_correct(item.type, status),
        evidence_hit=evidence_hit(
            gold_chunk_ids(chunks, item.evidence),
            [source.chunk_id for source in result.sources],
        ),
        cost_usd=result.cost_usd,
        latency_ms=result.retrieval_ms + result.generation_ms + result.support_ms,
        flagged_sentences=flagged,
        repaired=result.repair is not None and result.repair.outcome == "repaired",
        support_refused=status == "refused" and flagged > 0,
        trace_id=result.trace_id,
        error=error,
    )


def _mean(values: Iterable[float | bool | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return mean(present) if present else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(scores: list[ItemScore]) -> dict:
    answered = [s for s in scores if s.status is not None]
    answerable = [s for s in answered if s.type == "answerable"]
    latencies = [s.latency_ms for s in answered if s.latency_ms is not None]
    return {
        "questions": len(scores),
        "run_failures": len(scores) - len(answered),
        "judge_failures": sum(s.error is not None for s in answered),
        "correctness": _mean(s.correctness for s in scores),
        "correctness_by_type": {
            t: _mean(s.correctness for s in scores if s.type == t) for t in TYPES
        },
        "correctness_by_source": {
            source: _mean(s.correctness for s in scores if s.source == source)
            for source in sorted({s.source for s in scores})
        },
        "refusal_accuracy": _mean(
            s.refusal_ok for s in answered if s.type == "unanswerable"
        ),
        "false_refusal_rate": _mean(s.status == "refused" for s in answerable),
        "evidence_recall": _mean(s.evidence_hit for s in answered),
        "evidence_unknown": sum(
            s.evidence_hit is None and s.type != "unanswerable" for s in answered
        ),
        "token_f1": _mean(s.f1 for s in answerable),
        "invalid_rate": _mean(s.status == "invalid" for s in answered),
        "support_flagged_rate": _mean(s.flagged_sentences > 0 for s in answered),
        "support_repaired": sum(s.repaired for s in answered),
        "support_refusals": sum(s.support_refused for s in answered),
        "cost_usd": sum(s.cost_usd or 0.0 for s in answered),
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
