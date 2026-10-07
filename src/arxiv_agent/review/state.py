import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from arxiv_agent.llm import Usage


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
    via: Literal["search", "snowball", "reviewer"]
    # The sub-queries that returned it, or for snowballed papers the kept
    # papers it's linked to ("cited by 2406.07791", "cites 2406.07791").
    found_by: list[str]


class Score(BaseModel):
    arxiv_id: str
    score: int = Field(ge=0, le=10)
    reason: str


def _is_score(item: object) -> bool:
    try:
        Score.model_validate(item)
    except ValidationError:
        return False
    return True


class Scores(BaseModel):
    # What the Screener must return. A malformed entry (seen live: one with no
    # score) is dropped instead of failing the whole batch; that paper then
    # counts as not scored.
    scores: list[Score]

    @field_validator("scores", mode="before")
    @classmethod
    def drop_malformed(cls, items: object) -> object:
        if not isinstance(items, list):
            return items  # not a list at all: let validation report it
        return [item for item in items if _is_score(item)]


class ScreenedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    published: str
    via: Literal["search", "snowball", "reviewer"]
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
    # skipped: the time budget ran out before the Reader got to it
    source: Literal["full_text", "abstract_only", "unavailable", "skipped"]
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


class Budget(BaseModel):
    # Limits for one review, checked between steps: a step can overshoot by its
    # own size (each call's max_tokens bounds that). About 3x a measured run
    # (36 LLM calls, 48k tokens, 4 min), so they stop runaways, not normal runs.
    model_config = ConfigDict(extra="forbid")

    max_llm_calls: int = 100
    max_tokens: int = 150_000
    max_seconds: float = 600

    def problem(self, spent: Usage, elapsed: float) -> str | None:
        if spent.llm_calls >= self.max_llm_calls:
            return f"used {spent.llm_calls} of {self.max_llm_calls} LLM calls"
        if spent.tokens >= self.max_tokens:
            return f"used {spent.tokens:,} of {self.max_tokens:,} tokens"
        if elapsed >= self.max_seconds:
            return f"ran {elapsed:.0f} of {self.max_seconds:.0f} seconds"
        return None


class ReviewDecision(BaseModel):
    # What the person sends back at the pause: arXiv ids to take out of the
    # kept list, and ids to put in (any arXiv paper, found by search or not).
    model_config = ConfigDict(extra="forbid")

    remove: list[str] = []
    add: list[str] = []


class Edit(BaseModel):
    # One change the reviewer made, and what it says about the pipeline. These
    # are labels for the eval set:
    #   removed                          -> the Screener kept a paper it shouldn't have
    #   added, found but dropped earlier -> the Screener dropped one it should have kept
    #   added, never found               -> search and snowballing missed it
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    action: Literal["removed", "added"]
    label: Literal["screener_false_positive", "screener_false_negative", "search_miss"]
    title: str
    screener_score: int | None  # None when search never found the paper


def elapsed(state: "ReviewState", now: float) -> float:
    # Running time only: time spent waiting for a person doesn't count.
    return now - state["started_at"] - state.get("paused_seconds", 0.0)


class ReviewState(TypedDict, total=False):
    # Typed fields, not a shared message list: each node reads only what it
    # needs, so a paper the Screener drops can never reach a later prompt.
    question: str
    budget: Budget
    started_at: float  # time.time() when the review started
    # Every node returns what its own LLM calls used; the reducer (operator.add)
    # sums them, so even parallel calls inside a node are all counted.
    spent: Annotated[Usage, operator.add]
    stopped: str  # why the review stopped early, if the budget ran out
    pause_for_review: bool  # deep mode: wait for a person after screening
    # YYYY-MM-DD: only papers first submitted before this date are found, as if
    # the review ran then (the survey eval uses it to compare fairly)
    published_before: str
    review_requested_at: float  # when the wait began
    paused_seconds: float  # total time spent waiting, left out of the time budget
    sub_queries: list[str]
    criteria: list[str]
    candidates: list[Candidate]
    kept: list[ScreenedPaper]
    dropped: list[ScreenedPaper]
    snowballed: bool
    edits: list[Edit]  # what the reviewer changed at the pause
    ignored_edits: list[str]  # edits that couldn't be applied, and why
    claims: list[Claim]
    read: list[ReadReport]
    draft: str
    drafts: int  # how many drafts the Synthesizer has written
    critique: Critique
    review: str  # the final text, citing [arXiv:ID]
    references: list[ScreenedPaper]  # the cited papers, in order of first citation
    evidence: list[Evidence]  # every sentence -> the claims and quotes behind it
    removed: list[str]  # sentences the Critic still rejected after the last draft
