"""Repair or remove: what should happen to a sentence the support check fails?

6.3b removed it, and on the Q&A eval that once cost a correct answer: the
sentence was true apart from one detail, and nothing else was left. 6.3c asks
the answer model for one rewrite first. This eval runs both on the same
failing answers, so only the action differs.

Cases: real answers from the Answerer to the Q&A eval's answerable questions
(all sentences supported), with one supported sentence broken by code, as in
6.3: a number changed, "not" inserted, or the claim stretched to every model,
dataset and setting. Answers the check failed on its own are kept as "real"
cases. The check runs once per case; then three actions on its verdicts:
- none: the broken answer as it is (what no check ships)
- remove: drop the failing sentences (6.3b)
- repair: one rewrite, re-checked, then drop what still fails (6.3c)
Scored per action:
- break gone: the text code inserted (the new number, the "not", the
  "Across all ..." prefix) is no longer in the answer
- correctness: the eval judge against the gold answers; the unbroken answer is
  graded too, as the ceiling
- others kept: the answer's other sentences still there word for word
The cases quote answers built from paper text, so they live in data/.
"""

from collections.abc import Iterable
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.evals.judge_eval import (
    AUXILIARY,
    OVERGENERAL_PREFIX,
    change_number,
    negate,
    overgeneralize,
)
from arxiv_agent.qa.support import SentenceSupport
from arxiv_agent.review.citations import split_sentences
from arxiv_agent.review.numbers import result_numbers

Kind = Literal["real", "number_changed", "negated", "overgeneralized"]
BROKEN_KINDS: tuple[Kind, ...] = ("number_changed", "negated", "overgeneralized")
ACTIONS = ("none", "remove", "repair")
NEGATION_PREFIX = "It is not the case that"


class RepairCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    item_id: str
    kind: Kind
    question: str
    gold_answers: list[str]
    arxiv_id: str
    version: int
    chunk_ids: list[str]  # the sources, in [S#] order
    original: str  # the Answerer's answer
    broken: str  # what the check sees (real cases: the original)
    sentence: str  # the sentence that was broken ("" for real cases)
    broken_sentence: str
    marker: str | None  # what code inserted; must not survive (None: real)


def sentences(text: str) -> list[str]:
    # Split the way the support checker does.
    return [s for paragraph in text.split("\n") for s in split_sentences(paragraph)]


def _break(kind: Kind, sentence: str) -> tuple[str, str] | None:
    # The broken sentence and its marker: text that is only there because of
    # the break, so finding it in a final answer means the break survived.
    if kind == "number_changed":
        broken = change_number(sentence)
        if broken is None:
            return None
        start = result_numbers(sentence)[0].start(1)
        number_and_word = broken[start:].split()[:2]
        if len(number_and_word) == 2 and number_and_word[1].startswith("["):
            number_and_word = number_and_word[:1]
        return broken, " ".join(number_and_word)
    if kind == "negated":
        broken = negate(sentence)
        if broken.startswith(NEGATION_PREFIX):
            return broken, NEGATION_PREFIX
        auxiliary = AUXILIARY.search(sentence).group(0)
        return broken, f"{auxiliary} not"
    return overgeneralize(sentence), OVERGENERAL_PREFIX.rstrip(", ")


def break_answer(
    answer: str, support: list[SentenceSupport]
) -> list[tuple[Kind, str, str, str, str]]:
    # One case per kind: the first supported sentence each break applies to.
    # Returns (kind, broken answer, sentence, broken sentence, marker).
    cases = []
    for kind in BROKEN_KINDS:
        for s in support:
            if s.verdict != "supported":
                continue
            made = _break(kind, s.sentence)
            if made is None:
                continue
            broken_sentence, marker = made
            if marker.lower() in answer.lower() or answer.count(s.sentence) != 1:
                continue  # the marker must be new, the sentence findable
            broken = answer.replace(s.sentence, broken_sentence)
            cases.append((kind, broken, s.sentence, broken_sentence, marker))
            break
    return cases


def break_gone(case: RepairCase, text: str) -> bool | None:
    if case.marker is None:
        return None
    return case.marker.lower() not in text.lower()


def others_kept(case: RepairCase, text: str) -> float | None:
    if case.kind == "real":
        return None  # which sentences were fine is the judge's call there
    others = [s for s in sentences(case.original) if s != case.sentence]
    if not others:
        return None
    return sum(s in text for s in others) / len(others)


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    status: str
    break_gone: bool | None
    others_kept: float | None
    correctness: float | None  # None: the judge failed


class RepairResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Kind
    flagged: bool  # the check failed at least one sentence
    flagged_break: bool | None  # it failed the broken sentence (None: real)
    false_flags: int  # supported sentences it failed (broken cases only)
    original: float | None  # correctness of the unbroken answer
    actions: dict[str, Outcome]  # none / remove / repair
    repair_outcome: str | None = None  # repaired / fallback; None: not needed
    repair_reason: str | None = None
    repair_ms: float | None = None
    error: str | None = None


def _share(values: Iterable[bool | float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return mean(present) if present else None


def summarize(results: list[RepairResult]) -> dict:
    scored = [r for r in results if r.error is None]
    broken = [r for r in scored if r.kind != "real"]
    real = [r for r in scored if r.kind == "real"]
    repaired = [r for r in scored if r.repair_outcome is not None]

    def actions(rs: list[RepairResult]) -> dict:
        return {
            action: {
                "break_gone": _share(r.actions[action].break_gone for r in rs),
                "correctness": _share(r.actions[action].correctness for r in rs),
                "refused": _share(r.actions[action].status == "refused" for r in rs),
                "others_kept": _share(r.actions[action].others_kept for r in rs),
            }
            for action in ACTIONS
        }

    return {
        "cases": len(results),
        "errors": len(results) - len(scored),
        "broken": {
            "cases": len(broken),
            "flagged_break": _share(r.flagged_break for r in broken),
            "false_flags": sum(r.false_flags for r in broken),
            "original_correctness": _share(r.original for r in broken),
            "all": actions(broken),
            # Where the check fired, the only place remove and repair differ.
            "flagged": actions([r for r in broken if r.flagged]),
            "by_kind": {
                kind: actions([r for r in broken if r.kind == kind])
                for kind in BROKEN_KINDS
            },
        },
        "real": {
            "cases": len(real),
            "original_correctness": _share(r.original for r in real),
            "all": actions(real),
        },
        "repair": {
            "used": len(repaired),
            "fallbacks": sum(r.repair_outcome == "fallback" for r in repaired),
            "ms_p50": median(r.repair_ms for r in repaired) if repaired else None,
        },
    }
