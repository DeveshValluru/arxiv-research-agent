from typing import TypedDict

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
    found_by: list[str]  # the sub-queries that returned it


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
    score: int
    reason: str


class ReviewState(TypedDict, total=False):
    # Typed fields, not a shared message list: each node reads only what it
    # needs, so a paper the Screener drops can never reach a later node.
    question: str
    sub_queries: list[str]
    criteria: list[str]
    candidates: list[Candidate]
    kept: list[ScreenedPaper]
    dropped: list[ScreenedPaper]
