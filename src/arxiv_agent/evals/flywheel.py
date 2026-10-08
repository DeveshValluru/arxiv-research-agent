"""The flywheel: failures seen in real use become eval cases (7.3).

Three places record that something went wrong, and each feeds an eval:
- Q&A answers a guard flagged (Langfuse, environment "development"): an
  invalid answer, or sentences the support check failed. A person writes the
  right answer -> evals/qa_flagged.jsonl, which the Q&A eval and the CI gate
  pick up like any other set.
- Sentences the review's Critic removed (finished review jobs in Postgres): a
  person says whether the Critic was right -> data/evals/judge_flagged.jsonl,
  real cases for the judge eval (6.3), which so far had only sentences broken
  by code. In data/, not git: they quote the cited passages.
- Reviewer edits (review_labels, 5.3): papers the reviewer added (a search
  miss, a screener false negative) or removed (a screener false positive) ->
  evals/review_labels.jsonl, review eval cases for the same question, as of
  the day it was asked.
Nothing becomes a case until a person accepts it (scripts/triage_flags.py): a
guard's flag is a lead, not a label.
"""

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.evals.judge_eval import SupportCase
from arxiv_agent.evals.review_eval import SurveyCase
from arxiv_agent.evals.runner import EvalItem, ItemType

CANDIDATES = Path("data/evals/flag_candidates.jsonl")
Kind = Literal["qa", "critic", "label"]
Status = Literal["pending", "accepted", "rejected"]
SHOULD_KEEP = {"search_miss", "screener_false_negative"}
SHOULD_DROP = {"screener_false_positive"}
JUDGED = {"overstated", "unsupported"}  # the Critic's judge said so
CITATIONS = re.compile(r"\s*\[(?:K|S)\d+(?:\s*[,;]\s*(?:K|S)\d+)*\]")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # stable, so harvesting twice doesn't duplicate
    kind: Kind
    found: str  # when it happened (ISO date)
    reason: str  # why it was flagged
    question: str
    arxiv_id: str | None = None  # qa: the paper asked about; label: the paper
    version: int | None = None
    answer: str | None = None  # qa: what was answered
    sentence: str | None = None  # critic: the sentence it removed
    sources: list[str] = []  # critic: the arXiv ids of the cited passages
    passages: list[str] = []  # critic: the cited passages
    label: str | None = None  # label: search_miss / screener_false_...
    title: str | None = None
    trace_id: str | None = None
    job_id: str | None = None
    status: Status = "pending"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def qa_candidate(trace: dict) -> Candidate | None:
    # trace: the "ask" root and its support-check span, as harvest_flags.py
    # reads them from Langfuse: trace_id, time, question, paper ("2411.15594v6"),
    # answer, status, problems, failed (sentences the support check failed).
    reasons = []
    if trace.get("status") == "invalid":
        reasons.append("invalid answer: " + "; ".join(trace.get("problems", [])))
    if trace.get("failed"):
        reasons.append(f"support check failed {len(trace['failed'])} sentence(s)")
    if not reasons:
        return None
    arxiv_id, _, version = trace["paper"].rpartition("v")
    return Candidate(
        id=f"qa:{trace['trace_id']}",
        kind="qa",
        found=trace["time"][:10],
        reason="; ".join(reasons),
        question=trace["question"],
        arxiv_id=arxiv_id,
        version=int(version),
        answer=trace.get("answer"),
        trace_id=trace["trace_id"],
    )


def _plain(sentence: str) -> str:
    return " ".join(CITATIONS.sub("", sentence).split()).rstrip(".")


def critic_candidates(
    job_id: str, created: str, trace_id: str | None, result: dict
) -> list[Candidate]:
    # A finished review's removed sentences, with the Critic's last verdict on
    # each and the passages of the claims it cited.
    checks = {
        _plain(check["sentence"]): check
        for check in (result.get("critique") or {}).get("checks", [])
    }
    claims = {claim["label"]: claim for claim in result.get("claims", [])}
    found = []
    for sentence in result.get("removed", []):
        check = checks.get(_plain(sentence))
        if check is None or check["verdict"] not in JUDGED:
            # No judge verdict to confirm: "uncited" is the writer's mistake
            # (code catches it), not something to grade the judge on.
            continue
        cited = [claims[label] for label in check["labels"] if label in claims]
        found.append(
            Candidate(
                id=f"critic:{job_id}:{_hash(_plain(sentence))}",
                kind="critic",
                found=created[:10],
                reason=f"{check['verdict']}: {check['reason']}",
                question=result.get("question", ""),
                sentence=sentence,
                sources=[claim.get("arxiv_id", "") for claim in cited],
                passages=[claim["passage"] for claim in cited],
                trace_id=trace_id,
                job_id=job_id,
            )
        )
    return found


def label_candidate(row: dict) -> Candidate:
    # A review_labels row, with the job's created_at.
    return Candidate(
        id=f"label:{row['job_id']}:{row['arxiv_id']}",
        kind="label",
        found=str(row["created_at"])[:10],
        reason=row["label"],
        question=row["question"],
        arxiv_id=row["arxiv_id"],
        label=row["label"],
        title=row["title"],
        job_id=str(row["job_id"]),
    )


def load_candidates(path: Path = CANDIDATES) -> list[Candidate]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Candidate.model_validate_json(line) for line in lines if line.strip()]


def save_candidates(candidates: list[Candidate], path: Path = CANDIDATES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(c.model_dump_json() + "\n" for c in candidates),
        encoding="utf-8",
        newline="\n",
    )


def merge(existing: list[Candidate], found: list[Candidate]) -> list[Candidate]:
    # Keep what was already triaged; add only what's new.
    known = {candidate.id for candidate in existing}
    return [*existing, *(c for c in found if c.id not in known)]


def qa_item(
    candidate: Candidate, item_type: ItemType, gold_answers: list[str]
) -> EvalItem:
    return EvalItem(
        id=f"flagged-{_hash(candidate.id)}",
        source="flagged",
        arxiv_id=candidate.arxiv_id,
        version=candidate.version,
        question=candidate.question,
        type=item_type,
        answer_type=None if item_type == "unanswerable" else "abstractive",
        gold_answers=gold_answers,
        evidence=[],
        evidence_matched=False,
    )


def judge_case(candidate: Candidate, supported: bool) -> SupportCase:
    # The person's call, not the Critic's, is the right answer.
    return SupportCase(
        id=f"flagged-{_hash(candidate.id)}",
        kind="flagged",
        supported=supported,
        sentence=_plain(candidate.sentence) + ".",
        arxiv_id=candidate.sources[0] if candidate.sources else "",
        section="",
        passage="\n\n".join(candidate.passages),
    )


def review_case(labels: list[Candidate]) -> SurveyCase:
    # One case per review: everything the reviewer added is gold, everything
    # they removed must not be kept. Runs as of the day it was asked.
    first = labels[0]
    return SurveyCase(
        id=f"labels-{first.job_id[:8]}",
        survey="",
        version=0,
        title=first.question,
        cutoff=min(label.found for label in labels),
        question=first.question,
        references=0,
        with_arxiv_id=0,
        gold=sorted({c.arxiv_id for c in labels if c.label in SHOULD_KEEP}),
        exclude=sorted({c.arxiv_id for c in labels if c.label in SHOULD_DROP}),
    )
