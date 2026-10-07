import threading

import pytest
from langgraph.errors import GraphRecursionError

from arxiv_agent.llm import Usage
from arxiv_agent.review.state import Budget
from tests.review_fakes import (
    AGREEMENT,
    CALL_USAGE,
    FOLLOWUP,
    JUDGELM,
    POSITION,
    PREJUDICE,
    SWAP,
    FakeArxiv,
    FakeClock,
    FakeLibrary,
    FakeOpenAlex,
    chat,
    faithful_reader,
    judge,
    run,
)

# Two sentences; the strict judge calls the first one overstated.
BAD_DRAFT = "All LLM judges favour the first answer [K1]. Order can flip verdicts [K2]."


def linked() -> dict:
    # Snowballing adds PREJUDICE and FOLLOWUP, so four papers are kept.
    return {
        "arxiv": FakeArxiv(
            bibliographies={
                POSITION.arxiv_id: [PREJUDICE.arxiv_id, JUDGELM.arxiv_id],
                SWAP.arxiv_id: [PREJUDICE.arxiv_id, AGREEMENT.arxiv_id],
            }
        ),
        "openalex": FakeOpenAlex(citing={POSITION.arxiv_id: [FOLLOWUP.arxiv_id]}),
    }


# --- the budget itself -------------------------------------------------------------


@pytest.mark.parametrize(
    ("spent", "elapsed", "problem"),
    [
        (Usage(llm_calls=9, prompt_tokens=900), 59, None),
        (Usage(llm_calls=10), 0, "used 10 of 10 LLM calls"),
        (
            Usage(llm_calls=1, prompt_tokens=800, completion_tokens=200),
            0,
            "used 1,000 of 1,000 tokens",
        ),
        (Usage(), 60, "ran 60 of 60 seconds"),
    ],
)
def test_budget_problem(spent, elapsed, problem):
    budget = Budget(max_llm_calls=10, max_tokens=1_000, max_seconds=60)
    assert budget.problem(spent, elapsed) == problem


def test_usage_adds_up():
    total = sum([CALL_USAGE, CALL_USAGE, Usage(cost_usd=0.002)], Usage())
    assert total == Usage(
        llm_calls=2, prompt_tokens=200, completion_tokens=20, cost_usd=0.002
    )
    assert total.tokens == 220


# --- accounting ----------------------------------------------------------------


def test_every_llm_call_in_the_review_is_counted():
    writer, checker = chat(), judge()

    state, _ = run(writer, checker, **linked())

    # Writer: planner, 2 screenings, 4 papers read, 1 draft. Judge: 2 sentences.
    calls = len(writer.calls) + len(checker.calls)
    assert (len(writer.calls), len(checker.calls)) == (8, 2)
    assert state["spent"].llm_calls == calls
    assert state["spent"].tokens == calls * CALL_USAGE.tokens
    assert "stopped" not in state


# --- stopping early, keeping what's done ---------------------------------------------


def test_out_of_calls_after_screening_keeps_the_paper_list():
    writer, library = chat(), FakeLibrary()

    state, _ = run(writer, library=library, budget=Budget(max_llm_calls=2))

    assert state["stopped"] == "used 2 of 2 LLM calls"  # planner + screener
    assert [p.arxiv_id for p in state["kept"]] == [POSITION.arxiv_id, SWAP.arxiv_id]
    assert library.ingested == []
    assert writer.prompts("reader-llm") == []
    assert "review" not in state


def test_time_running_out_while_reading_skips_the_remaining_papers():
    clock = FakeClock()

    def slow_download(arxiv_id):
        clock.now += 100

    writer = chat()
    state, _ = run(
        writer,
        library=FakeLibrary(on_ingest=slow_download),
        budget=Budget(max_seconds=150),
        clock=clock,
        **linked(),
    )

    assert [r.source for r in state["read"]] == [
        "full_text",
        "full_text",
        "skipped",
        "skipped",
    ]
    assert len(state["claims"]) == 2  # the two papers that were read
    assert state["stopped"] == "ran 200 of 150 seconds"
    assert writer.prompts("synthesizer-llm") == []


def test_the_budget_ends_a_rewrite_loop_and_ships_only_checked_sentences():
    writer = chat(**{"synthesizer-llm": BAD_DRAFT})

    # 7 calls reach the first critique (planner, screener, 2 readers, writer,
    # 2 judge calls); one rewrite brings it to 10, and the budget stops it.
    state, _ = run(writer, max_revisions=10, budget=Budget(max_llm_calls=10))

    assert state["drafts"] == 2
    assert state["stopped"] == "used 10 of 10 LLM calls"
    assert state["removed"] == ["All LLM judges favour the first answer [K1]."]
    assert state["review"] == "Order can flip verdicts [arXiv:2305.17926]."


def test_the_recursion_limit_is_the_backstop():
    # With a huge budget and rewrite allowance, LangGraph's own step limit
    # still ends a loop that never converges.
    writer = chat(**{"synthesizer-llm": BAD_DRAFT})
    with pytest.raises(GraphRecursionError):
        run(writer, max_revisions=100, budget=Budget(max_llm_calls=10_000))


# --- overlap --------------------------------------------------------------------


def test_the_reader_extracts_a_paper_while_the_next_one_downloads():
    extracting = threading.Event()
    overlapped = []

    def reader(messages):
        extracting.set()
        return faithful_reader(messages)

    def download(arxiv_id):
        if arxiv_id == SWAP.arxiv_id:  # the second paper
            overlapped.append(extracting.wait(timeout=5))

    state, _ = run(
        chat(**{"reader-llm": reader}), library=FakeLibrary(on_ingest=download)
    )

    # Paper 1's extraction started while paper 2 was still downloading.
    assert overlapped == [True]
    assert len(state["claims"]) == 2
