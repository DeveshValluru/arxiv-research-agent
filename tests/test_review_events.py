import json

from arxiv_agent.review.events import (
    describe_event,
    format_report,
    format_review_request,
    result_of,
    step_summary,
)
from tests.review_fakes import POSITION, SWAP, chat, run


def run_with_events(**kwargs):
    events = []
    state, _ = run(
        chat(), on_event=lambda kind, data: events.append((kind, data)), **kwargs
    )
    return state, events


def test_every_step_is_reported_in_order():
    _, events = run_with_events()

    assert [data["step"] for kind, data in events if kind == "step"] == [
        "start",
        "planner",
        "searcher",
        "screener",
        "snowball",
        "screener",
        "reader",
        "synthesizer",
        "critic",
        "finalize",
    ]


def test_the_reader_reports_each_paper():
    _, events = run_with_events()

    assert [data for kind, data in events if kind == "progress"] == [
        {"done": 1, "total": 2, "paper": POSITION.arxiv_id, "source": "full_text"},
        {"done": 2, "total": 2, "paper": SWAP.arxiv_id, "source": "full_text"},
    ]


def test_events_carry_counts_not_data():
    _, events = run_with_events()

    steps = {data["step"]: data for kind, data in events if kind == "step"}
    assert steps["searcher"] == {"step": "searcher", "candidates": 3}
    assert steps["critic"] == {"step": "critic", "verdict": "pass", "problems": 0}
    assert len(json.dumps(events)) < 2_000  # small enough to store and stream


def test_step_summary_handles_an_empty_update():
    assert step_summary("stop", None) == {"step": "stop"}


def test_the_result_is_plain_json_without_the_intermediate_state():
    state, _ = run_with_events()

    result = result_of(state)

    assert json.loads(json.dumps(result)) == result
    assert "candidates" not in result and "draft" not in result
    assert result["found"] == {"search": 3}
    assert result["references"][0]["arxiv_id"] == POSITION.arxiv_id


def test_the_report_reads_a_stored_result():
    state, _ = run_with_events()

    report = format_report(json.loads(json.dumps(result_of(state))))

    assert "3 papers from search, 0 from snowballing" in report
    assert state["review"] in report
    assert (
        f"[arXiv:{SWAP.arxiv_id}] {SWAP.title}. Ada Lovelace, Alan Turing et al., 2024"
        in report
    )


def test_describe_event():
    assert describe_event("status", {"status": "running", "attempt": 2}) == (
        "status: running (attempt 2)"
    )
    assert describe_event("step", {"step": "screener", "kept": 8, "dropped": 28}) == (
        "screener: kept 8, dropped 28"
    )
    assert describe_event("step", {"step": "planner", "searches": ["a b", "c"]}) == (
        "planner: searches a b; c"
    )
    assert describe_event(
        "step", {"step": "stop", "stopped": "ran 73 of 60 seconds"}
    ) == ("stop: ran 73 of 60 seconds")
    assert (
        describe_event(
            "progress",
            {"done": 3, "total": 8, "paper": "2406.07791", "source": "full_text"},
        )
        == "reading 3/8: 2406.07791 (full_text)"
    )


def test_describe_event_shows_the_reviewers_edits():
    edit = {
        "action": "removed",
        "arxiv_id": "2305.17926",
        "label": "screener_false_positive",
    }
    data = {"step": "human_review", "kept": 2, "dropped": 1, "edits": [edit]}
    assert describe_event("step", data) == (
        "human_review: kept 2, dropped 1, edits removed 2305.17926"
    )


def test_the_review_request_shows_what_will_and_wont_be_read():
    card = {
        "arxiv_id": "2406.07791",
        "title": "Judging the Judges",
        "score": 9,
        "reason": "Measures position bias.",
        "via": "search",
    }
    text = format_review_request(
        {"question": "q?", "kept": [card], "dropped": [card] * 20}, dropped_shown=2
    )
    assert "Kept, will be read (1):" in text
    assert "Dropped (top 2 of 20):" in text
    assert text.count("Measures position bias.") == 3
