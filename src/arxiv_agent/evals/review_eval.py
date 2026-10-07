"""Scoring a literature review against a survey paper's bibliography.

A survey's references are papers experts chose for a topic: a gold set nobody
here made up. Each review runs "as of" the survey's first version date (only
papers submitted before it can be found), so it's compared with what the
experts could have cited, not penalized for papers that came out later.

Only references with an arXiv id are in the gold set (half to three quarters
of them). Classics cited by their venue version (MT-Bench as NeurIPS) are
invisible to it, so recall here is a lower bound.

The funnel, stage by stage:
- search recall: the share of the gold set the searches returned
- candidate recall: the same after snowballing
- kept precision: the share of kept papers the experts cited. A lower bound
  too: a survey can't cite every relevant paper
- screener lift: kept precision / the gold share among all candidates. Above
  1, the Screener picks expert-cited papers better than chance
- cited precision: the share of the papers the review cites that experts cited
- support rate: the Critic's verdicts on the final draft's sentences
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from arxiv_agent.review.state import ReviewState


class SurveyCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    survey: str  # arXiv id of the survey paper
    version: int  # the version whose bibliography is the gold set
    title: str
    cutoff: str  # YYYY-MM-DD, the survey's first version: reviews run as of then
    question: str
    references: int  # entries in the bibliography
    with_arxiv_id: int
    gold: list[str]  # cited arXiv papers submitted before the cutoff


class ReviewScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    gold: int
    candidates: int = 0
    search_recall: float | None = None
    candidate_recall: float | None = None
    kept: int = 0
    kept_gold: int = 0
    kept_precision: float | None = None
    candidate_precision: float | None = None
    screener_lift: float | None = None
    cited: int = 0
    cited_gold: int = 0
    cited_precision: float | None = None
    sentences: int = 0
    support_rate: float | None = None
    removed_sentences: int = 0
    gold_kept: list[str] = []  # which expert-cited papers the review kept
    llm_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    stopped: str | None = None
    error: str | None = None
    trace_id: str | None = None


def load_cases(path: Path) -> list[SurveyCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SurveyCase.model_validate_json(line) for line in lines if line.strip()]


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def score_review(
    case: SurveyCase, state: ReviewState, seconds: float, trace_id: str | None
) -> ReviewScore:
    gold = set(case.gold)
    candidates = state.get("candidates", [])
    searched = {c.arxiv_id for c in candidates if c.via == "search"}
    found = {c.arxiv_id for c in candidates}
    kept = [paper.arxiv_id for paper in state.get("kept", [])]
    cited = [paper.arxiv_id for paper in state.get("references", [])]
    checks = state["critique"].checks if "critique" in state else []
    judged = [check for check in checks if check.verdict != "unchecked"]

    kept_precision = _share(len(gold.intersection(kept)), len(kept))
    candidate_precision = _share(len(found & gold), len(found))
    lift = None
    if kept_precision is not None and candidate_precision:
        lift = kept_precision / candidate_precision
    spent = state.get("spent")
    return ReviewScore(
        id=case.id,
        gold=len(gold),
        candidates=len(found),
        search_recall=_share(len(searched & gold), len(gold)),
        candidate_recall=_share(len(found & gold), len(gold)),
        kept=len(kept),
        kept_gold=len(gold.intersection(kept)),
        kept_precision=kept_precision,
        candidate_precision=candidate_precision,
        screener_lift=lift,
        cited=len(cited),
        cited_gold=len(gold.intersection(cited)),
        cited_precision=_share(len(gold.intersection(cited)), len(cited)),
        sentences=len(checks),
        support_rate=_share(sum(c.verdict == "supported" for c in judged), len(judged)),
        removed_sentences=len(state.get("removed", [])),
        gold_kept=[arxiv_id for arxiv_id in kept if arxiv_id in gold],
        llm_calls=spent.llm_calls if spent else 0,
        tokens=spent.tokens if spent else 0,
        seconds=round(seconds, 1),
        stopped=state.get("stopped"),
        trace_id=trace_id,
    )


METRICS = (
    "search_recall",
    "candidate_recall",
    "kept_precision",
    "screener_lift",
    "cited_precision",
    "support_rate",
)


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def summarize(scores: list[ReviewScore]) -> dict:
    # Means over the reviews that ran; a failed one is counted, not averaged in.
    ran = [score for score in scores if score.error is None]
    return {
        "cases": len(scores),
        "failures": len(scores) - len(ran),
        **{metric: _mean(getattr(s, metric) for s in ran) for metric in METRICS},
        "llm_calls": sum(s.llm_calls for s in ran),
        "tokens": sum(s.tokens for s in ran),
        "seconds": round(sum(s.seconds for s in ran), 1),
    }
