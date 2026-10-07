"""The literature-review graph (5.1a: Planner -> Searcher -> Screener).

5.1b adds Reader -> Synthesizer -> Citation Critic and the rewrite loop.
"""

from langfuse import Langfuse, get_client
from langgraph.graph import END, START, StateGraph

from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.state import ReviewState


def build_review_graph(nodes: ReviewNodes):
    graph = StateGraph(ReviewState)
    graph.add_node("planner", nodes.plan)
    graph.add_node("searcher", nodes.search)
    graph.add_node("screener", nodes.screen)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "screener")
    graph.add_edge("screener", END)
    return graph.compile()


async def run_review(
    question: str, nodes: ReviewNodes, langfuse: Langfuse | None = None
) -> tuple[ReviewState, str | None]:
    # One trace per review: every node, LLM call and tool call nests under it.
    langfuse = langfuse or get_client()
    with langfuse.start_as_current_observation(
        as_type="agent", name="literature-review", input={"question": question}
    ) as root:
        state = await build_review_graph(nodes).ainvoke({"question": question})
        root.update(
            output={
                "kept": [paper.arxiv_id for paper in state["kept"]],
                "candidates": len(state["candidates"]),
            }
        )
        trace_id = root.trace_id
    return state, None if trace_id == "0" * 32 else trace_id
