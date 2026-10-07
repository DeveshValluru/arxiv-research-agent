"""The literature-review graph.

    planner -> searcher -> screener -> snowball -> screener -> reader
            -> synthesizer -> critic -> (synthesizer again | finalize)

Two loops, both bounded: Snowball sends its finds back through the Screener
once, and the Critic sends a draft back to the Synthesizer at most
max_revisions times. The routing functions only read state; the decisions
behind them (what's kept, the verdict) were made by the nodes.
"""

from langfuse import Langfuse, get_client
from langgraph.graph import END, START, StateGraph

from arxiv_agent.review.citations import finalize
from arxiv_agent.review.critic import Critic
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.reader import Reader
from arxiv_agent.review.state import ReviewState
from arxiv_agent.review.writer import Synthesizer


def after_screening(state: ReviewState) -> str:
    if not state["kept"]:
        return END  # nothing relevant found: nothing to read or write
    return "reader" if state.get("snowballed") else "snowball"


def after_reading(state: ReviewState) -> str:
    return "synthesizer" if state["claims"] else END


def after_critique(state: ReviewState) -> str:
    return "synthesizer" if state["critique"].verdict == "revise" else "finalize"


def build_review_graph(
    finder: ReviewNodes, reader: Reader, synthesizer: Synthesizer, critic: Critic
):
    graph = StateGraph(ReviewState)
    graph.add_node("planner", finder.plan)
    graph.add_node("searcher", finder.search)
    graph.add_node("screener", finder.screen)
    graph.add_node("snowball", finder.snowball)
    graph.add_node("reader", reader.read)
    graph.add_node("synthesizer", synthesizer.write)
    graph.add_node("critic", critic.check)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "screener")
    graph.add_conditional_edges(
        "screener", after_screening, ["snowball", "reader", END]
    )
    graph.add_edge("snowball", "screener")
    graph.add_conditional_edges("reader", after_reading, ["synthesizer", END])
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", after_critique, ["synthesizer", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_review(
    question: str, graph, langfuse: Langfuse | None = None
) -> tuple[ReviewState, str | None]:
    # One trace per review: every node, LLM call and tool call nests under it.
    langfuse = langfuse or get_client()
    with langfuse.start_as_current_observation(
        as_type="agent", name="literature-review", input={"question": question}
    ) as root:
        state = await graph.ainvoke({"question": question})
        critique = state.get("critique")
        root.update(
            output={
                "candidates": len(state["candidates"]),
                "kept": [paper.arxiv_id for paper in state["kept"]],
                "claims": len(state.get("claims", [])),
                "drafts": state.get("drafts", 0),
                "verdict": critique.verdict if critique else None,
                "removed": len(state.get("removed", [])),
                "review": state.get("review"),
            },
            level="WARNING" if state.get("removed") else None,
        )
        trace_id = root.trace_id
    return state, None if trace_id == "0" * 32 else trace_id
