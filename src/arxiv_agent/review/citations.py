"""Reading the Synthesizer's draft: sentences, claim labels, and the final text.

The Synthesizer cites claim labels ([K3]), never arXiv ids. Code turns each
label into the id of the paper the claim came from, so the model can't garble
an id or cite a paper from memory: the error is impossible, not just detected.
"""

import re

from pydantic import BaseModel, ConfigDict

from arxiv_agent.guardrails.output import Blocked, OutputGuard
from arxiv_agent.review.numbers import missing_numbers
from arxiv_agent.review.state import (
    PASSING,
    Claim,
    Evidence,
    ReviewState,
    SentenceCheck,
    Verdict,
)

LABEL = re.compile(r"K\d+")
LABEL_GROUP = re.compile(r"\[(K\d+(?:\s*[,;]\s*K\d+)*)\]")  # [K1] or [K1, K2]
CITATION_RUN = re.compile(r"(?:\s*\[K\d+(?:\s*[,;]\s*K\d+)*\])+")  # [K1][K2]
# [K1] in reviews, [S1] in Q&A answers
LEADING_CITATIONS = re.compile(r"^(?:\[[KS]\d+(?:\s*[,;]\s*[KS]\d+)*\]\s*)+")
# An id or link the model wrote itself instead of citing a label.
FROM_MEMORY = re.compile(
    r"arXiv:\s*\d|https?://|www\.|\b\d{4}\.\d{4,5}\b", re.IGNORECASE
)
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])")
ABBREVIATIONS = {"e.g.", "i.e.", "al.", "vs.", "cf.", "etc.", "fig.", "eq.", "sec."}


class Sentence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraph: int
    text: str
    labels: list[str]


def split_sentences(paragraph: str) -> list[str]:
    pieces, start = [], 0
    for match in BOUNDARY.finditer(paragraph):
        piece = paragraph[start : match.start()]
        last_word = piece.split()[-1].lower().lstrip("([")
        if last_word in ABBREVIATIONS:  # "e.g. GPT-4" isn't a sentence end
            continue
        pieces.append(piece.strip())
        start = match.end()
    pieces.append(paragraph[start:].strip())

    # A citation written after the full stop belongs to the sentence before.
    sentences: list[str] = []
    for piece in pieces:
        lead = LEADING_CITATIONS.match(piece)
        if lead and sentences:
            sentences[-1] += " " + lead.group().strip()
            piece = piece[lead.end() :].strip()
        if piece:
            sentences.append(piece)
    return sentences


def labels_in(text: str) -> list[str]:
    found = [
        label for group in LABEL_GROUP.findall(text) for label in LABEL.findall(group)
    ]
    return list(dict.fromkeys(found))


def parse_draft(draft: str) -> list[Sentence]:
    # Paragraphs are kept so the final text can be put back together; headings
    # and blank lines are dropped.
    sentences = []
    blocks = [block for block in re.split(r"\n\s*\n", draft) if block.strip()]
    for index, block in enumerate(blocks):
        lines = [
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        ]
        text = " ".join(" ".join(lines).split())
        if not text:
            continue
        for sentence in split_sentences(text):
            sentences.append(
                Sentence(paragraph=index, text=sentence, labels=labels_in(sentence))
            )
    return sentences


def code_check(sentence: Sentence, claims: dict[str, Claim]) -> SentenceCheck | None:
    # Deterministic checks: free, instant, and never wrong in the same way
    # twice. None means the sentence passed and still needs the judge.
    def fail(verdict: Verdict, reason: str) -> SentenceCheck:
        return SentenceCheck(
            paragraph=sentence.paragraph,
            sentence=sentence.text,
            labels=sentence.labels,
            verdict=verdict,
            reason=reason,
        )

    if not sentence.labels:
        return fail("uncited", "it cites no claim; cite the claims it relies on")
    unknown = [label for label in sentence.labels if label not in claims]
    if unknown:
        return fail("bad_citation", f"{', '.join(unknown)} isn't one of the claims")
    if FROM_MEMORY.search(LABEL_GROUP.sub("", sentence.text)):
        return fail(
            "bad_citation", "it writes an id or link itself; cite claim labels only"
        )
    # A result number must come from the evidence. The judge eval caught the
    # judge passing 98.0% against a passage that says 98.2%.
    missing = missing_numbers(
        LABEL_GROUP.sub("", sentence.text),
        [claims[label].passage for label in sentence.labels],
    )
    if missing:
        return fail(
            "unsupported", f"{', '.join(missing)} isn't in the passages it cites"
        )
    return None


def render(sentence: str, claims: dict[str, Claim]) -> str:
    # [K1][K4] -> [arXiv:2406.07791, arXiv:2305.17926], one entry per paper.
    def cite(match: re.Match) -> str:
        ids = dict.fromkeys(
            claims[label].arxiv_id
            for label in LABEL.findall(match.group())
            if label in claims
        )
        return " [" + ", ".join(f"arXiv:{i}" for i in ids) + "]" if ids else ""

    return CITATION_RUN.sub(cite, sentence)


def finalize(state: ReviewState, guard: OutputGuard | None = None) -> ReviewState:
    # The review keeps only sentences the Critic accepted: after the last
    # allowed rewrite, a rejected sentence is removed rather than shipped.
    # Then the output guard (layer 4) checks each one as it will be shown: a
    # sentence that fails is blocked, never shipped.
    claims = {claim.label: claim for claim in state["claims"]}
    papers = {paper.arxiv_id: paper for paper in state["kept"]}
    paragraphs: dict[int, list[str]] = {}
    evidence, removed, cited = [], [], []
    blocked: list[Blocked] = []
    for check in state["critique"].checks:
        if check.verdict not in PASSING:
            removed.append(check.sentence)
            continue
        text = render(check.sentence, claims)
        violations = guard.check(text, allowed_ids=set(papers)) if guard else []
        if violations:
            blocked.append(Blocked(sentence=text, violations=violations))
            continue
        paragraphs.setdefault(check.paragraph, []).append(text)
        used = [claims[label] for label in check.labels]
        evidence.append(Evidence(sentence=text, claims=used))
        for claim in used:
            if claim.arxiv_id not in cited:
                cited.append(claim.arxiv_id)
    return {
        "review": "\n\n".join(" ".join(p) for p in paragraphs.values()),
        "references": [papers[arxiv_id] for arxiv_id in cited],
        "evidence": evidence,
        "removed": removed,
        "blocked": blocked,
    }


def describe(check: SentenceCheck) -> str:
    return f'- "{check.sentence}" -> {check.verdict}: {check.reason}'
