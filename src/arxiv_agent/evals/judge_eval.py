"""Judging the judge: does the Critic catch sentences we know are wrong?

The Critic's judge said 98-100% of review sentences were supported (5.4). That
is either good writing or a lenient judge, and the judge can't grade itself.
So the eval uses sentences whose answer we know: real sentences from papers in
the index (supported by the passage they come from), and copies of them
broken in known ways by code, never by a model:
- number_changed: one number altered (62% -> 69%)
- negated: "not" inserted
- overgeneralized: claimed for every model, dataset and setting
- wrong_passage: shown with a passage from a different paper
Every broken copy should be judged not supported. The production prompt and
judge are used as they are (review/critic.py).

The cases quote paper text, so they're built from the local index into data/
(not committed); the builder is deterministic.
"""

import asyncio
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.llm import (
    ChatModel,
    LLMOutputError,
    LLMUnavailableError,
    parse_json_object,
)
from arxiv_agent.review.critic import CRITIC_PROMPT, judging_request
from arxiv_agent.review.numbers import missing_numbers, result_numbers
from arxiv_agent.review.state import Claim, Judgment

Kind = Literal[
    "supported", "number_changed", "negated", "overgeneralized", "wrong_passage"
]
KINDS: tuple[Kind, ...] = (
    "supported",
    "number_changed",
    "negated",
    "overgeneralized",
    "wrong_passage",
)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
AUXILIARY = re.compile(
    r"\b(is|are|was|were|can|could|does|do|did|has|have|had|will|would|should|may|might)\b"
)
OVERGENERAL_PREFIX = "Across all models, datasets and settings, "
RESULT_VERB = re.compile(
    r"\b(?:achiev|outperform|improv|reach|obtain|yield|reduc|increas|decreas|drop|"
    r"agree|score|surpass|exceed|gain|boost|lower|rais|prefer|correlat)\w*",
    re.IGNORECASE,
)


class SupportCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Kind
    supported: bool  # the right answer
    sentence: str
    arxiv_id: str  # the paper the evidence comes from
    section: str
    passage: str


def change_number(sentence: str) -> str | None:
    numbers = result_numbers(sentence)
    if not numbers:
        return None
    match = numbers[0]
    text = match.group(1)
    if "." in text:
        decimals = len(text.split(".")[1])
        new = f"{float(text) + 0.17 * (1 if float(text) < 50 else -1):.{decimals}f}"
    elif "," in text:
        new = f"{int(text.replace(',', '')) * 2:,}"
    else:
        value = int(text)
        new = str(value + 7 if value < 50 else value - 13)
    return sentence[: match.start(1)] + new + sentence[match.end(1) :]


def negate(sentence: str) -> str:
    # "not" after the first auxiliary verb; otherwise deny the whole sentence.
    match = AUXILIARY.search(sentence)
    if match and not sentence[match.end() :].lstrip().startswith("not"):
        return f"{sentence[: match.end()]} not{sentence[match.end() :]}"
    return f"It is not the case that {sentence[0].lower()}{sentence[1:]}"


def overgeneralize(sentence: str) -> str:
    return f"{OVERGENERAL_PREFIX}{sentence[0].lower()}{sentence[1:]}"


def candidate_sentences(text: str) -> list[str]:
    # Sentences worth testing: a result (a result verb and a result-like
    # number), plain prose, 10 to 40 words. Description ("Twitter is a huge
    # service with 500 million tweets") makes a poor test: overgeneralizing it
    # is just nonsense.
    picked = []
    for line in text.split("\n"):
        for sentence in SENTENCE.split(line.strip()):
            words = sentence.split()
            if not 10 <= len(words) <= 40 or not sentence.endswith("."):
                continue
            if any(mark in sentence for mark in ("$", "\\", "|", "/", "http", "www")):
                continue  # math, tables, links, paths
            if RESULT_VERB.search(sentence) and result_numbers(sentence):
                picked.append(sentence)
    return picked


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    chunk_id: str
    section: str
    sentence: str
    passage: str


def build_cases(bases: list[Base]) -> list[SupportCase]:
    # Five cases per base sentence. The wrong passage comes from the next base
    # that belongs to a different paper.
    cases = []
    for n, base in enumerate(bases):
        other = next(
            b for b in bases[n + 1 :] + bases[:n] if b.arxiv_id != base.arxiv_id
        )
        variants = {
            "supported": (base.sentence, base),
            "number_changed": (change_number(base.sentence), base),
            "negated": (negate(base.sentence), base),
            "overgeneralized": (overgeneralize(base.sentence), base),
            "wrong_passage": (base.sentence, other),
        }
        for kind, (sentence, evidence) in variants.items():
            if sentence is None:
                continue
            cases.append(
                SupportCase(
                    id=f"{base.chunk_id}:{kind}",
                    kind=kind,
                    supported=kind == "supported",
                    sentence=sentence,
                    arxiv_id=evidence.arxiv_id,
                    section=evidence.section,
                    passage=evidence.passage,
                )
            )
    return cases


class JudgedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Kind
    supported: bool
    verdict: str  # supported / overstated / unsupported, or "error"
    reason: str


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def summarize(judged: Iterable[JudgedCase]) -> dict:
    # "Caught" = a broken sentence not judged supported; "false alarm" = a true
    # sentence judged anything else. Errors are counted, not scored.
    judged = list(judged)
    scored = [j for j in judged if j.verdict != "error"]
    broken = [j for j in scored if not j.supported]
    true = [j for j in scored if j.supported]
    by_kind = {}
    for kind in KINDS:
        cases = [j for j in scored if j.kind == kind]
        right = sum((j.verdict == "supported") == j.supported for j in cases)
        by_kind[kind] = {"cases": len(cases), "correct": _share(right, len(cases))}
    return {
        "cases": len(judged),
        "errors": len(judged) - len(scored),
        "caught": _share(sum(j.verdict != "supported" for j in broken), len(broken)),
        "false_alarms": _share(sum(j.verdict != "supported" for j in true), len(true)),
        "accuracy": _share(
            sum((j.verdict == "supported") == j.supported for j in scored), len(scored)
        ),
        "by_kind": by_kind,
    }


async def judge_case(
    judge: ChatModel,
    case: SupportCase,
    limit: asyncio.Semaphore,
    prompt: str = CRITIC_PROMPT,
    number_check: bool = False,
) -> JudgedCase:
    # Exactly what the Critic sends: the production prompt, and the sentence
    # citing one claim whose passage is the evidence.
    evidence = Claim(
        label="K1",
        arxiv_id=case.arxiv_id,
        version=1,
        chunk_id=f"{case.arxiv_id}:evidence",
        section=case.section,
        claim="",
        quote="",
        passage=case.passage,
    )
    # number_check: the Critic's free code check first; the judge only sees
    # sentences that pass it.
    missing = missing_numbers(case.sentence, [case.passage]) if number_check else []
    if missing:
        return JudgedCase(
            id=case.id,
            kind=case.kind,
            supported=case.supported,
            verdict="unsupported",
            reason=f"code: {', '.join(missing)} not in the cited passage",
        )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": judging_request(f"{case.sentence[:-1]} [K1].", [evidence]),
        },
    ]
    async with limit:
        try:
            reply = await judge.complete(messages, name="judge-eval", max_tokens=150)
            judgment = parse_json_object(reply.text, Judgment)
            verdict, reason = judgment.verdict, judgment.reason
        except (LLMOutputError, LLMUnavailableError) as exc:
            verdict, reason = "error", f"{type(exc).__name__}: {exc}"[:200]
    return JudgedCase(
        id=case.id,
        kind=case.kind,
        supported=case.supported,
        verdict=verdict,
        reason=reason,
    )
