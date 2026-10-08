"""The pause after screening: a person checks the paper list before it's read.

Reading is the slow, expensive step, so the pause sits right before it: a bad
paper list costs you seconds instead of minutes of reading. While the graph
waits, its state sits in the checkpointer; nothing has to keep running.

How LangGraph pauses: a node calls interrupt(request). The graph saves its
state and stops. Resuming with Command(resume=decision) runs that node again
from the top, and this time interrupt() returns the decision.
"""

import time
from collections.abc import Callable

from langfuse import Langfuse, get_client
from langgraph.types import interrupt

from arxiv_agent.clients.arxiv import ARXIV_ID_PATTERN
from arxiv_agent.review.nodes import MAX_LOOKUP, VERSION_SUFFIX
from arxiv_agent.review.state import (
    Edit,
    ReviewDecision,
    ReviewState,
    ScreenedPaper,
)
from arxiv_agent.tools.toolbox import McpToolbox


def _card(paper: ScreenedPaper, flags: list[str]) -> dict:
    card = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "score": paper.score,
        "reason": paper.reason,
        "via": paper.via,
    }
    if flags:  # the person should know a paper tried to steer the Screener
        card["flags"] = flags
    return card


def review_request(state: ReviewState) -> dict:
    # What the person sees, as plain JSON: it's stored on the job and shown by
    # the CLI (later the UI).
    flags: dict[str, list[str]] = {}
    for flag in state.get("content_flags", []):
        flags.setdefault(flag.arxiv_id, []).append(f"{flag.reason}: {flag.text[:80]}")
    return {
        "question": state["question"],
        "kept": [_card(p, flags.get(p.arxiv_id, [])) for p in state["kept"]],
        "dropped": [_card(p, flags.get(p.arxiv_id, [])) for p in state["dropped"]],
    }


class HumanReview:
    def __init__(
        self,
        toolbox: McpToolbox,
        clock: Callable[[], float] = time.time,
        langfuse: Langfuse | None = None,
    ) -> None:
        self._toolbox = toolbox
        self._clock = clock
        self._langfuse = langfuse or get_client()

    def request(self, state: ReviewState) -> ReviewState:
        # A step of its own, so the time the wait began is checkpointed: the
        # review step can't keep it, because it starts over when it resumes.
        return {"review_requested_at": self._clock()}

    async def review(self, state: ReviewState) -> ReviewState:
        decision = ReviewDecision.model_validate(interrupt(review_request(state)))
        waited = self._clock() - state["review_requested_at"]
        with self._langfuse.start_as_current_observation(
            as_type="chain", name="human-review", input=decision.model_dump()
        ) as span:
            kept, dropped, edits, ignored = await self._apply(state, decision)
            span.update(
                output={
                    "kept": [paper.arxiv_id for paper in kept],
                    "edits": [edit.model_dump() for edit in edits],
                    "ignored": ignored,
                    "waited_seconds": round(waited),
                }
            )
        return {
            "kept": kept,
            "dropped": dropped,
            "edits": edits,
            "ignored_edits": ignored,
            "paused_seconds": state.get("paused_seconds", 0.0) + waited,
        }

    async def _apply(
        self, state: ReviewState, decision: ReviewDecision
    ) -> tuple[list[ScreenedPaper], list[ScreenedPaper], list[Edit], list[str]]:
        # The person decides; code checks each edit can be done, and says so
        # when one can't instead of failing the review.
        kept, dropped = list(state["kept"]), list(state["dropped"])
        edits: list[Edit] = []
        ignored: list[str] = []

        for raw in decision.remove:
            arxiv_id = VERSION_SUFFIX.sub("", raw.strip())
            paper = next((p for p in kept if p.arxiv_id == arxiv_id), None)
            if paper is None:
                ignored.append(f"remove {raw}: not in the kept list")
                continue
            kept.remove(paper)
            dropped.insert(
                0, paper.model_copy(update={"reason": "removed by the reviewer"})
            )
            edits.append(_edit(paper, "removed", "screener_false_positive"))

        to_look_up = []
        for raw in decision.add:
            arxiv_id = VERSION_SUFFIX.sub("", raw.strip())
            if not ARXIV_ID_PATTERN.match(arxiv_id):
                ignored.append(f"add {raw}: not an arXiv id")
            elif any(p.arxiv_id == arxiv_id for p in kept):
                ignored.append(f"add {raw}: already kept")
            elif paper := next((p for p in dropped if p.arxiv_id == arxiv_id), None):
                dropped.remove(paper)
                kept.append(
                    paper.model_copy(update={"reason": "added by the reviewer"})
                )
                edits.append(_edit(paper, "added", "screener_false_negative"))
            else:
                to_look_up.append(arxiv_id)

        if len(to_look_up) > MAX_LOOKUP:  # get_metadata's own limit
            ignored += [
                f"add {i}: more than {MAX_LOOKUP} new papers at once"
                for i in to_look_up[MAX_LOOKUP:]
            ]
            to_look_up = to_look_up[:MAX_LOOKUP]
        if to_look_up:
            found, problem = await self._look_up(to_look_up)
            for arxiv_id in to_look_up:
                info = found.get(arxiv_id)
                if info is None:
                    ignored.append(f"add {arxiv_id}: {problem or 'not on arXiv'}")
                    continue
                paper = ScreenedPaper(
                    arxiv_id=arxiv_id,
                    version=info["version"],
                    title=info["title"],
                    authors=info["authors"],
                    published=info["published"],
                    via="reviewer",
                    found_by=["added by the reviewer"],
                    score=10,
                    reason="added by the reviewer",
                )
                kept.append(paper)
                edits.append(
                    Edit(
                        arxiv_id=arxiv_id,
                        action="added",
                        label="search_miss",
                        title=paper.title,
                        screener_score=None,
                    )
                )
        return kept, dropped, edits, ignored

    async def _look_up(
        self, arxiv_ids: list[str]
    ) -> tuple[dict[str, dict], str | None]:
        # Papers search never found: the Reader needs their metadata.
        result = await self._toolbox.call("get_metadata", {"arxiv_ids": arxiv_ids})
        if not result.ok:
            return {}, f"couldn't look it up ({result.content})"
        return {paper["arxiv_id"]: paper for paper in result.content["papers"]}, None


def _edit(paper: ScreenedPaper, action: str, label: str) -> Edit:
    return Edit(
        arxiv_id=paper.arxiv_id,
        action=action,
        label=label,
        title=paper.title,
        screener_score=paper.score,
    )
