import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.guardrails.output import OutputGuard

REFUSAL = "The paper doesn't say."
CITATION = re.compile(r"\[S(\d+)\]")
THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class CheckedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    status: Literal["answered", "refused", "invalid"]
    cited: list[int]
    problems: list[str]


def check_answer(
    raw: str, n_sources: int, guard: OutputGuard | None = None
) -> CheckedAnswer:
    # The prompt asks for these rules; this enforces them. A refusal must be the
    # exact sentence (so it can be counted), and an answer must cite real sources.
    # The output guard adds links and prompt leaks (layer 4).
    text = THINKING.sub("", raw).strip()

    if text.replace("’", "'") == REFUSAL:
        return CheckedAnswer(text=text, status="refused", cited=[], problems=[])

    cited = list(dict.fromkeys(int(n) for n in CITATION.findall(text)))
    problems = [] if cited else ["no citations"]
    problems += [
        f"cites S{n}, but only S1-S{n_sources} exist"
        for n in cited
        if not 1 <= n <= n_sources
    ]
    if guard is not None:
        problems += [violation.detail for violation in guard.check(text)]

    return CheckedAnswer(
        text=text,
        status="invalid" if problems else "answered",
        cited=cited,
        problems=problems,
    )
