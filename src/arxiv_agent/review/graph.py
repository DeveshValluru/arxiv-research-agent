"""The literature-review graph.

    start -> planner -> searcher -> screener -> snowball -> screener
          -> [deep mode: request_review -> human_review (pause)] -> reader
          -> synthesizer -> critic -> (synthesizer again | finalize)

Loop control, in three layers:
1. Each loop has its own limit: Snowball runs once, and the Critic allows
   max_revisions rewrites.
2. The budget (LLM calls, tokens, seconds) is checked before each expensive
   step. When it's used up, the review stops and keeps what it has.
3. LangGraph's recursion limit is the backstop if routing ever goes wrong.
The routing functions only read state and the clock; they never call a model.

With a checkpointer, the state is saved after every step under a thread id
(a job's id). That's what lets a review pause for a person, and lets a review
whose worker died continue from its last finished step.
"""

import time
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

from langfuse import Langfuse, get_client, propagate_attributes
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from arxiv_agent.guardrails.output import OutputGuard
from arxiv_agent.llm import Usage
from arxiv_agent.review.citations import finalize
from arxiv_agent.review.critic import CRITIC_PROMPT, Critic
from arxiv_agent.review.events import step_summary
from arxiv_agent.review.human import HumanReview
from arxiv_agent.review.nodes import PLANNER_PROMPT, SCREENER_PROMPT, ReviewNodes
from arxiv_agent.review.reader import READER_PROMPT, Reader
from arxiv_agent.review.state import Budget, ReviewDecision, ReviewState, elapsed
from arxiv_agent.review.writer import SYNTHESIZER_PROMPT, Synthesizer

# What the output guard checks a review against for leaks: every prompt in it.
REVIEW_PROMPTS = (
    PLANNER_PROMPT,
    SCREENER_PROMPT,
    READER_PROMPT,
    SYNTHESIZER_PROMPT,
    CRITIC_PROMPT,
)
# The longest legitimate path is 17 steps (a pause and 2 rewrites); anything
# near 50 means the routing is looping, and LangGraph raises GraphRecursionError.
RECURSION_LIMIT = 50


class LoopControl:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock

    def start(self, state: ReviewState) -> ReviewState:
        return {
            "budget": state.get("budget") or Budget(),
            "started_at": self._clock(),
            "spent": Usage(),
        }

    def out_of_budget(self, state: ReviewState) -> str | None:
        return state["budget"].problem(state["spent"], elapsed(state, self._clock()))

    def stop(self, state: ReviewState) -> ReviewState:
        # Everything done so far is kept: the paper list, the claims, or a
        # checked draft (finalize then drops its failing sentences).
        return {"stopped": self.out_of_budget(state) or "the budget ran out"}

    def after_screening(self, state: ReviewState) -> str:
        if not state["kept"]:
            return END  # nothing relevant found: nothing to read or write
        if self.out_of_budget(state):
            return "stop"
        if not state.get("snowballed"):
            return "snowball"
        return "request_review" if state.get("pause_for_review") else "reader"

    def after_review(self, state: ReviewState) -> str:
        if not state["kept"]:
            return END  # the reviewer removed every paper
        return "stop" if self.out_of_budget(state) else "reader"

    def after_reading(self, state: ReviewState) -> str:
        # The budget first: the Reader may have skipped papers because of it.
        if self.out_of_budget(state):
            return "stop"
        return "synthesizer" if state["claims"] else END

    def after_critique(self, state: ReviewState) -> str:
        if state["critique"].verdict != "revise":
            return "finalize"
        return "stop" if self.out_of_budget(state) else "synthesizer"

    @staticmethod
    def after_stop(state: ReviewState) -> str:
        return "finalize" if "critique" in state else END


def build_review_graph(
    finder: ReviewNodes,
    reader: Reader,
    synthesizer: Synthesizer,
    critic: Critic,
    human: HumanReview,
    clock: Callable[[], float] = time.time,
    checkpointer: BaseCheckpointSaver | None = None,
):
    control = LoopControl(clock)
    graph = StateGraph(ReviewState)
    graph.add_node("start", control.start)
    graph.add_node("planner", finder.plan)
    graph.add_node("searcher", finder.search)
    graph.add_node("screener", finder.screen)
    graph.add_node("snowball", finder.snowball)
    graph.add_node("request_review", human.request)
    graph.add_node("human_review", human.review)
    graph.add_node("reader", reader.read)
    graph.add_node("synthesizer", synthesizer.write)
    graph.add_node("critic", critic.check)
    graph.add_node("finalize", partial(finalize, guard=OutputGuard(REVIEW_PROMPTS)))
    graph.add_node("stop", control.stop)

    graph.add_edge(START, "start")
    graph.add_edge("start", "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "screener")
    graph.add_conditional_edges(
        "screener",
        control.after_screening,
        ["snowball", "request_review", "reader", "stop", END],
    )
    graph.add_edge("snowball", "screener")
    graph.add_edge("request_review", "human_review")
    graph.add_conditional_edges(
        "human_review", control.after_review, ["reader", "stop", END]
    )
    graph.add_conditional_edges(
        "reader", control.after_reading, ["synthesizer", "stop", END]
    )
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic", control.after_critique, ["synthesizer", "finalize", "stop"]
    )
    graph.add_conditional_edges("stop", control.after_stop, ["finalize", END])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


EventHandler = Callable[[str, dict], None]


class ReviewRun(NamedTuple):
    state: ReviewState
    trace_id: str | None
    pause: dict | None  # what to show the person, when the review is waiting


async def run_review(
    question: str,
    graph,
    *,
    thread_id: str,
    budget: Budget | None = None,
    pause_for_review: bool = False,
    published_before: str | None = None,
    decision: dict | None = None,
    on_event: EventHandler | None = None,
    session_id: str | None = None,
    langfuse: Langfuse | None = None,
) -> ReviewRun:
    # Starts a review, or carries on with one under the same thread_id:
    # - waiting for a person and given a decision: resume with it;
    # - stopped mid-way (its worker died): continue from the last checkpoint;
    # - otherwise: start from the question.
    # One trace per run; the thread id is also the Langfuse session, so a
    # review's runs (before and after a pause) are grouped. An eval passes its
    # run id as session_id instead, to group all of its reviews.
    # on_event(kind, data) hears "step" after each node and "progress" from
    # inside long nodes; the graph streams them as it runs.
    langfuse = langfuse or get_client()
    on_event = on_event or (lambda kind, data: None)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    saved = await graph.aget_state(config) if graph.checkpointer else None
    if saved and saved.interrupts and decision is not None:
        # Validated here, so a bad decision fails before anything runs. It also
        # always has both keys: LangGraph ignores an empty (falsy) resume value
        # and would just pause again.
        graph_input = Command(
            resume=ReviewDecision.model_validate(decision).model_dump()
        )
    elif saved and saved.next:
        graph_input = None  # continue from the last checkpoint
    else:
        graph_input = {
            "question": question,
            "budget": budget or Budget(),
            "pause_for_review": pause_for_review,
        }
        if published_before:
            graph_input["published_before"] = published_before
    with (
        langfuse.start_as_current_observation(
            as_type="agent", name="literature-review", input={"question": question}
        ) as root,
        propagate_attributes(
            trace_name="literature-review", session_id=session_id or thread_id
        ),
    ):
        state: ReviewState = {}
        async for mode, chunk in graph.astream(
            graph_input, config=config, stream_mode=["updates", "custom", "values"]
        ):
            if mode == "values":
                state = chunk  # the whole state after each step; the last is final
            elif mode == "updates":
                for node, update in chunk.items():
                    if node != "__interrupt__":  # the pause is reported below
                        on_event("step", step_summary(node, update))
            else:
                on_event("progress", chunk)

        pause = None
        if graph.checkpointer:
            saved = await graph.aget_state(config)
            pause = saved.interrupts[0].value if saved.interrupts else None
        critique = state.get("critique")
        root.update(
            output={
                "paused_for_review": pause is not None,
                "candidates": len(state.get("candidates", [])),
                "kept": [paper.arxiv_id for paper in state.get("kept", [])],
                "edits": len(state.get("edits", [])),
                "claims": len(state.get("claims", [])),
                "drafts": state.get("drafts", 0),
                "verdict": critique.verdict if critique else None,
                "removed": len(state.get("removed", [])),
                "blocked": [b.model_dump() for b in state.get("blocked", [])],
                "stopped": state.get("stopped"),
                "spent": state["spent"].model_dump() if "spent" in state else None,
                "review": state.get("review"),
            },
            # Guard firings are flywheel signals: easy to find as warnings.
            level="WARNING"
            if state.get("removed") or state.get("stopped") or state.get("blocked")
            else None,
        )
        trace_id = root.trace_id
    return ReviewRun(state, None if trace_id == "0" * 32 else trace_id, pause)
