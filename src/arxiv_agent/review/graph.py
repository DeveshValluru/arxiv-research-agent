"""The literature-review graph.

    start -> planner -> searcher -> screener -> snowball -> screener -> reader
          -> synthesizer -> critic -> (synthesizer again | finalize)

Loop control, in three layers:
1. Each loop has its own limit: Snowball runs once, and the Critic allows
   max_revisions rewrites.
2. The budget (LLM calls, tokens, seconds) is checked before each expensive
   step. When it's used up, the review stops and keeps what it has.
3. LangGraph's recursion limit is the backstop if routing ever goes wrong.
The routing functions only read state and the clock; they never call a model.
"""

import time
from collections.abc import Callable

from langfuse import Langfuse, get_client, propagate_attributes
from langgraph.graph import END, START, StateGraph

from arxiv_agent.llm import Usage
from arxiv_agent.review.citations import finalize
from arxiv_agent.review.critic import Critic
from arxiv_agent.review.events import step_summary
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.reader import Reader
from arxiv_agent.review.state import Budget, ReviewState
from arxiv_agent.review.writer import Synthesizer

# The longest legitimate path is 15 steps (with 2 rewrites); anything near 50
# means the routing is looping, and LangGraph raises GraphRecursionError.
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
        elapsed = self._clock() - state["started_at"]
        return state["budget"].problem(state["spent"], elapsed)

    def stop(self, state: ReviewState) -> ReviewState:
        # Everything done so far is kept: the paper list, the claims, or a
        # checked draft (finalize then drops its failing sentences).
        return {"stopped": self.out_of_budget(state) or "the budget ran out"}

    def after_screening(self, state: ReviewState) -> str:
        if not state["kept"]:
            return END  # nothing relevant found: nothing to read or write
        if self.out_of_budget(state):
            return "stop"
        return "reader" if state.get("snowballed") else "snowball"

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
    clock: Callable[[], float] = time.time,
):
    control = LoopControl(clock)
    graph = StateGraph(ReviewState)
    graph.add_node("start", control.start)
    graph.add_node("planner", finder.plan)
    graph.add_node("searcher", finder.search)
    graph.add_node("screener", finder.screen)
    graph.add_node("snowball", finder.snowball)
    graph.add_node("reader", reader.read)
    graph.add_node("synthesizer", synthesizer.write)
    graph.add_node("critic", critic.check)
    graph.add_node("finalize", finalize)
    graph.add_node("stop", control.stop)

    graph.add_edge(START, "start")
    graph.add_edge("start", "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "screener")
    graph.add_conditional_edges(
        "screener", control.after_screening, ["snowball", "reader", "stop", END]
    )
    graph.add_edge("snowball", "screener")
    graph.add_conditional_edges(
        "reader", control.after_reading, ["synthesizer", "stop", END]
    )
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic", control.after_critique, ["synthesizer", "finalize", "stop"]
    )
    graph.add_conditional_edges("stop", control.after_stop, ["finalize", END])
    graph.add_edge("finalize", END)
    return graph.compile()


EventHandler = Callable[[str, dict], None]


async def run_review(
    question: str,
    graph,
    budget: Budget | None = None,
    on_event: EventHandler | None = None,
    session_id: str | None = None,
    langfuse: Langfuse | None = None,
) -> tuple[ReviewState, str | None]:
    # One trace per review: every node, LLM call and tool call nests under it.
    # session_id groups a job's traces (5.3: before and after a pause).
    # on_event(kind, data) hears "step" after each node and "progress" from
    # inside long nodes; the graph streams them as it runs.
    langfuse = langfuse or get_client()
    on_event = on_event or (lambda kind, data: None)
    with (
        langfuse.start_as_current_observation(
            as_type="agent", name="literature-review", input={"question": question}
        ) as root,
        propagate_attributes(trace_name="literature-review", session_id=session_id),
    ):
        state: ReviewState = {}
        async for mode, chunk in graph.astream(
            {"question": question, "budget": budget or Budget()},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode=["updates", "custom", "values"],
        ):
            if mode == "values":
                state = chunk  # the whole state after each step; the last is final
            elif mode == "updates":
                for node, update in chunk.items():
                    on_event("step", step_summary(node, update))
            else:
                on_event("progress", chunk)
        critique = state.get("critique")
        root.update(
            output={
                "candidates": len(state["candidates"]),
                "kept": [paper.arxiv_id for paper in state["kept"]],
                "claims": len(state.get("claims", [])),
                "drafts": state.get("drafts", 0),
                "verdict": critique.verdict if critique else None,
                "removed": len(state.get("removed", [])),
                "stopped": state.get("stopped"),
                "spent": state["spent"].model_dump(),
                "review": state.get("review"),
            },
            level="WARNING" if state.get("removed") or state.get("stopped") else None,
        )
        trace_id = root.trace_id
    return state, None if trace_id == "0" * 32 else trace_id
