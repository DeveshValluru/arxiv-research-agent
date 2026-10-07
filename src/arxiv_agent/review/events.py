"""What a review reports while it runs, and what it keeps when it's done.

Progress events carry counts and decisions, never the data itself: a review's
state runs to hundreds of KB, and a progress view needs "8 kept", not the
eight papers. The full result is stored once, at the end.
"""

from collections import Counter

from pydantic import BaseModel

from arxiv_agent.review.state import ReviewState

# The parts of the final state worth keeping. Candidates (every search result)
# and the raw draft are intermediate; the rendered review replaces the draft.
RESULT_FIELDS = (
    "question",
    "budget",
    "sub_queries",
    "criteria",
    "kept",
    "dropped",
    "read",
    "claims",
    "drafts",
    "critique",
    "review",
    "references",
    "evidence",
    "removed",
    "spent",
    "stopped",
)


def step_summary(node: str, update: dict | None) -> dict:
    # Keyed by what the step produced rather than by its name, so a new node
    # needs no change here.
    update = update or {}
    summary: dict = {"step": node}
    if "sub_queries" in update:
        summary["searches"] = update["sub_queries"]
    if "candidates" in update:
        summary["candidates"] = len(update["candidates"])
    if "kept" in update:
        summary["kept"] = len(update["kept"])
        summary["dropped"] = len(update["dropped"])
    if "claims" in update:
        summary["claims"] = len(update["claims"])
    if "drafts" in update:
        summary["draft"] = update["drafts"]
    if "critique" in update:
        summary["verdict"] = update["critique"].verdict
        summary["problems"] = len(update["critique"].problems)
    if "review" in update:
        summary["removed"] = len(update["removed"])
    if "stopped" in update:
        summary["stopped"] = update["stopped"]
    return summary


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def result_of(state: ReviewState) -> dict:
    result = {key: _jsonable(state[key]) for key in RESULT_FIELDS if key in state}
    if "candidates" in state:
        result["found"] = dict(Counter(c.via for c in state["candidates"]))
    return result


def describe_event(kind: str, data: dict) -> str:
    # One line per event, for terminals and logs.
    if kind == "status":
        details = [f"{k} {v}" for k, v in data.items() if k != "status"]
        return f"status: {data['status']}" + (
            f" ({', '.join(details)})" if details else ""
        )
    if kind == "progress":
        return (
            f"reading {data['done']}/{data['total']}: "
            f"{data['paper']} ({data['source']})"
        )
    parts = []
    for key, value in data.items():
        text = "; ".join(value) if isinstance(value, list) else str(value)
        if key == "stopped":  # already a sentence: "used 6 of 2 LLM calls"
            parts.append(text)
        elif key != "step":
            parts.append(f"{key} {text}")
    return f"{data['step']}: {', '.join(parts)}" if parts else data["step"]


def format_report(result: dict) -> str:
    # The final report, from a stored result (or result_of(state)).
    found, kept = result.get("found", {}), result.get("kept", [])
    lines = [
        (
            f"{found.get('search', 0)} papers from search, "
            f"{found.get('snowball', 0)} from snowballing"
        ),
        f"{len(kept)} kept:",
        *(
            f"  {p['score']:2}  {p['arxiv_id']}  [{p['via']}]  {p['title'][:70]}"
            for p in kept
        ),
    ]
    if "read" in result:
        sources = dict(Counter(r["source"] for r in result["read"]))
        rejected = sum(len(r["rejected"]) for r in result["read"])
        lines.append(
            f"\nRead: {sources}; {len(result['claims'])} claims "
            f"({rejected} rejected by the quote check)"
        )
    spent = result["spent"]
    tokens = spent["prompt_tokens"] + spent["completion_tokens"]
    lines.append(
        f"Spent: {spent['llm_calls']} LLM calls, {tokens:,} tokens, "
        f"at least ${spent['cost_usd']:.4f} (not every provider reports a cost)"
    )
    if "stopped" in result:
        lines.append(f"STOPPED EARLY: {result['stopped']}; showing what was done")
    if "review" not in result:
        lines.append(
            "\nNo review: nothing relevant was found or read, or time ran out."
        )
        return "\n".join(lines)

    critique = result["critique"]
    checks = dict(Counter(check["verdict"] for check in critique["checks"]))
    lines += [
        (
            f"Drafts: {result['drafts']}, final check: {checks}, "
            f"verdict: {critique['verdict']}\n"
        ),
        result["review"],
        "\nReferences",
    ]
    for paper in result["references"]:
        authors = ", ".join(paper["authors"][:2])
        if len(paper["authors"]) > 2:
            authors += " et al."
        lines.append(
            f"  [arXiv:{paper['arxiv_id']}] {paper['title']}. "
            f"{authors}, {paper['published'][:4]}"
        )
    lines += [f"\nREMOVED (still failed the check): {s}" for s in result["removed"]]
    return "\n".join(lines)
