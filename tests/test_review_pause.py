import warnings

import pytest
from pydantic import ValidationError

from arxiv_agent.review.checkpoints import ThreadedPostgresSaver, serializer
from arxiv_agent.review.state import Budget, ReviewDecision
from tests.review_fakes import (
    AGREEMENT,
    FOLLOWUP,
    POSITION,
    SWAP,
    FakeClock,
    FakeLibrary,
    Harness,
    chat,
)

# The default fakes keep POSITION (9) and SWAP (7) and drop AGREEMENT (4).
# FOLLOWUP is on arXiv, but search never finds it.
DECISION = {
    "remove": [SWAP.arxiv_id],
    "add": [AGREEMENT.arxiv_id, f"{FOLLOWUP.arxiv_id}v1", "9999.99999", "not-an-id"],
}


def ids(papers) -> list[str]:
    return [paper.arxiv_id for paper in papers]


def test_deep_mode_pauses_after_screening_before_anything_is_read():
    harness = Harness(chat())

    paused = harness.run(pause_for_review=True)

    request = paused.pause
    assert [p["arxiv_id"] for p in request["kept"]] == [
        POSITION.arxiv_id,
        SWAP.arxiv_id,
    ]
    assert request["kept"][0] == {
        "arxiv_id": POSITION.arxiv_id,
        "title": POSITION.title,
        "score": 9,
        "reason": "Measures position bias.",
        "via": "search",
    }
    assert [p["arxiv_id"] for p in request["dropped"]] == [AGREEMENT.arxiv_id]
    assert harness.library.ingested == []  # the expensive part hasn't started
    assert harness.writer.prompts("reader-llm") == []
    assert "claims" not in paused.state


def test_quick_mode_doesnt_pause():
    review = Harness(chat()).run()
    assert review.pause is None
    assert "review" in review.state


def test_resuming_applies_the_reviewers_edits_and_labels_them():
    harness = Harness(chat())
    harness.run(pause_for_review=True)

    done = harness.run(decision=DECISION)

    state = done.state
    assert done.pause is None
    assert ids(state["kept"]) == [
        POSITION.arxiv_id,
        AGREEMENT.arxiv_id,
        FOLLOWUP.arxiv_id,
    ]
    assert [
        (e.arxiv_id, e.action, e.label, e.screener_score) for e in state["edits"]
    ] == [
        (SWAP.arxiv_id, "removed", "screener_false_positive", 7),
        (AGREEMENT.arxiv_id, "added", "screener_false_negative", 4),
        (FOLLOWUP.arxiv_id, "added", "search_miss", None),
    ]
    assert state["ignored_edits"] == [
        "add not-an-id: not an arXiv id",
        "add 9999.99999: not on arXiv",
    ]
    assert harness.library.ingested == ids(state["kept"])  # read what the person chose
    assert "review" in state
    # It continued from the checkpoint: planning and screening weren't redone.
    assert len(harness.writer.prompts("planner-llm")) == 1


def test_a_paper_the_reviewer_removed_never_reaches_a_later_prompt():
    harness = Harness(chat())
    harness.run(pause_for_review=True)

    harness.run(decision={"remove": [SWAP.arxiv_id]})

    later = [
        message["content"]
        for name in ("reader-llm", "synthesizer-llm")
        for messages in harness.writer.prompts(name)
        for message in messages
    ]
    assert not any(SWAP.arxiv_id in text or SWAP.title in text for text in later)


def test_removing_every_paper_ends_the_review():
    harness = Harness(chat())
    harness.run(pause_for_review=True)

    done = harness.run(decision={"remove": [POSITION.arxiv_id, SWAP.arxiv_id]})

    assert done.state["kept"] == []
    assert harness.library.ingested == []
    assert "review" not in done.state


def test_time_spent_waiting_doesnt_count_against_the_budget():
    clock = FakeClock()
    harness = Harness(chat(), clock=clock)
    harness.run(pause_for_review=True, budget=Budget(max_seconds=600))

    clock.now += 10_000  # the reviewer answers the next day
    done = harness.run(decision={})

    assert done.state["paused_seconds"] == 10_000
    assert "stopped" not in done.state
    assert "review" in done.state


def test_a_review_continues_from_its_last_checkpoint_after_a_crash():
    crashed = []

    def crash_once(arxiv_id):
        if not crashed:
            crashed.append(arxiv_id)
            raise RuntimeError("the worker died")

    harness = Harness(chat(), library=FakeLibrary(on_ingest=crash_once))
    with pytest.raises(RuntimeError, match="the worker died"):
        harness.run()

    done = harness.run()  # no decision, no question: carry on

    assert "review" in done.state
    # Planner and screener ran once; only the Reader step started over.
    assert len(harness.writer.prompts("planner-llm")) == 1
    assert len(harness.writer.prompts("screener-llm")) == 1


def test_a_decision_must_have_the_expected_shape():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate({"remove": [], "approve": True})


def test_a_paused_review_survives_a_restart_with_postgres(connect):
    # Two savers on two connections: the second plays a restarted worker.
    def postgres_saver():
        saver = ThreadedPostgresSaver(connect(), serde=serializer())
        saver.setup()
        return saver

    before = Harness(chat(), checkpointer=postgres_saver())
    assert before.run(pause_for_review=True).pause is not None

    after = Harness(chat(), checkpointer=postgres_saver())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        done = after.run(decision={"remove": [SWAP.arxiv_id]})

    assert ids(done.state["kept"]) == [POSITION.arxiv_id]
    assert "review" in done.state
    assert after.writer.prompts("planner-llm") == []  # nothing redone
    # Every type in the state is on the checkpoint allowlist.
    assert not [w for w in caught if "unregistered type" in str(w.message)]
