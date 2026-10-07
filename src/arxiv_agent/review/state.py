from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class Plan(BaseModel):
    # What the Planner must return. The limits are enforced on its output.
    sub_queries: list[str] = Field(min_length=2, max_length=6)
    criteria: list[str] = Field(min_length=1, max_length=5)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    published: str
    abstract: str
    via: Literal["search", "snowball"]
    # The sub-queries that returned it, or for snowballed papers the kept
    # papers it's linked to ("cited by 2406.07791", "cites 2406.07791").
    found_by: list[str]


class Score(BaseModel):
    arxiv_id: str
    score: int = Field(ge=0, le=10)
    reason: str


class Scores(BaseModel):
    # What the Screener must return.
    scores: list[Score]


class ScreenedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    published: str
    via: Literal["search", "snowball"]
    found_by: list[str]
    score: int
    reason: str


class ExtractedClaim(BaseModel):
    # What the Reader's model must return for each claim.
    claim: str
    passage: str  # the passage label it comes from, e.g. "P2"
    quote: str


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str  # "K1": what the Synthesizer cites; code turns it into an arXiv id
    arxiv_id: str
    version: int
    chunk_id: str
    section: str
    claim: str
    quote: str  # checked by code to appear in the passage
    passage: str  # the full passage, for the Critic


class ReadReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    source: Literal["full_text", "abstract_only", "unavailable"]
    claims: int
    rejected: list[str]  # why extracted claims were thrown away


Verdict = Literal[
    "supported",  # the cited passages say this
    "overstated",  # they support a narrower or weaker version
    "unsupported",  # they don't say this
    "uncited",  # no claim label (code check)
    "bad_citation",  # an unknown label, or an id/URL written from memory (code check)
    "unchecked",  # the judge failed; the draft isn't blamed for that
]
PASSING: set[str] = {"supported", "unchecked"}


class Judgment(BaseModel):
    # What the Critic's judge model must return for one sentence.
    verdict: Literal["supported", "overstated", "unsupported"]
    reason: str


class SentenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraph: int
    sentence: str
    labels: list[str]
    verdict: Verdict
    reason: str


class Critique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["pass", "revise", "give_up"]
    checks: list[SentenceCheck]

    @property
    def problems(self) -> list[SentenceCheck]:
        return [check for check in self.checks if check.verdict not in PASSING]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence: str  # as rendered in the review, with [arXiv:...] citations
    claims: list[Claim]


class ReviewState(TypedDict, total=False):
    # Typed fields, not a shared message list: each node reads only what it
    # needs, so a paper the Screener drops can never reach a later prompt.
    question: str
    sub_queries: list[str]
    criteria: list[str]
    candidates: list[Candidate]
    kept: list[ScreenedPaper]
    dropped: list[ScreenedPaper]
    snowballed: bool
    claims: list[Claim]
    read: list[ReadReport]
    draft: str
    drafts: int  # how many drafts the Synthesizer has written
    critique: Critique
    review: str  # the final text, citing [arXiv:ID]
    references: list[ScreenedPaper]  # the cited papers, in order of first citation
    evidence: list[Evidence]  # every sentence -> the claims and quotes behind it
    removed: list[str]  # sentences the Critic still rejected after the last draft
