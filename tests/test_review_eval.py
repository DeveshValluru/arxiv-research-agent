from datetime import date
from pathlib import Path

import pytest

from arxiv_agent.evals.review_eval import (
    ReviewScore,
    SurveyCase,
    load_cases,
    score_review,
    summarize,
)
from arxiv_agent.mcp_servers.arxiv_server import build_search_query
from tests.review_fakes import (
    AGREEMENT,
    FOLLOWUP,
    JUDGELM,
    POSITION,
    PREJUDICE,
    SWAP,
    FakeArxiv,
    FakeOpenAlex,
    Harness,
    chat,
)

EVAL_SET = Path(__file__).parent.parent / "evals" / "review_surveys.jsonl"


def linked() -> dict:
    # Search finds POSITION, SWAP, AGREEMENT; snowballing adds PREJUDICE and
    # JUDGELM (cited by kept papers) and FOLLOWUP (cites one, published 2026).
    return {
        "arxiv": FakeArxiv(
            bibliographies={
                POSITION.arxiv_id: [PREJUDICE.arxiv_id, JUDGELM.arxiv_id],
                SWAP.arxiv_id: [PREJUDICE.arxiv_id],
            }
        ),
        "openalex": FakeOpenAlex(citing={POSITION.arxiv_id: [FOLLOWUP.arxiv_id]}),
    }


def case(gold: list[str]) -> SurveyCase:
    return SurveyCase(
        id="judges",
        survey="2411.15594",
        version=6,
        title="A Survey on LLM-as-a-Judge",
        cutoff="2024-11-23",
        question="How biased are LLM judges?",
        references=226,
        with_arxiv_id=112,
        gold=gold,
    )


# --- reviewing "as of" a date ------------------------------------------------------


def test_the_search_query_can_end_at_a_date():
    assert build_search_query("judge bias", published_before=date(2024, 11, 23)) == (
        "all:judge AND all:bias AND submittedDate:[199101010000 TO 202411230000]"
    )


def test_a_review_as_of_a_date_searches_and_snowballs_only_older_papers():
    harness = Harness(chat(), **linked())

    review = harness.run(published_before="2025-01-01")

    assert all(
        "submittedDate:[199101010000 TO 202501010000]" in q
        for q in harness.arxiv.queries
    )
    found = [c.arxiv_id for c in review.state["candidates"]]
    assert PREJUDICE.arxiv_id in found
    assert FOLLOWUP.arxiv_id not in found  # it cites a kept paper, but from 2026


# --- scoring ----------------------------------------------------------------------


def test_a_review_is_scored_stage_by_stage():
    review = Harness(chat(), **linked()).run()
    gold = [POSITION.arxiv_id, AGREEMENT.arxiv_id, PREJUDICE.arxiv_id, "2001.00001"]

    score = score_review(case(gold), review.state, seconds=12.3, trace_id="t")

    # Kept: PREJUDICE (10), POSITION (9), SWAP (7), FOLLOWUP (6). The review
    # cites K1 and K2: PREJUDICE and POSITION.
    assert (score.gold, score.candidates) == (4, 6)
    assert score.search_recall == 2 / 4  # POSITION, AGREEMENT
    assert score.candidate_recall == 3 / 4  # + PREJUDICE from snowballing
    assert (score.kept_gold, score.kept, score.kept_precision) == (2, 4, 0.5)
    assert score.candidate_precision == 3 / 6
    assert score.screener_lift == 1.0
    assert (score.cited_gold, score.cited, score.cited_precision) == (2, 2, 1.0)
    assert score.gold_kept == [PREJUDICE.arxiv_id, POSITION.arxiv_id]
    assert (score.sentences, score.support_rate) == (2, 1.0)
    assert score.llm_calls == review.state["spent"].llm_calls
    assert (score.seconds, score.trace_id) == (12.3, "t")


def test_a_review_that_stopped_early_has_no_precision_to_score():
    review = Harness(chat(**{"reader-llm": '{"claims": []}'})).run()

    score = score_review(case([POSITION.arxiv_id]), review.state, 1.0, None)

    assert score.kept_precision == 0.5  # the paper list still counts
    assert (score.cited, score.cited_precision, score.support_rate) == (0, None, None)


def test_the_summary_averages_reviews_that_ran_and_counts_failures():
    scores = [
        ReviewScore(
            id="a", gold=10, search_recall=0.2, kept_precision=0.5, llm_calls=30
        ),
        ReviewScore(
            id="b", gold=10, search_recall=0.4, kept_precision=None, llm_calls=20
        ),
        ReviewScore(
            id="c", gold=10, error="LLMUnavailableError: every provider failed"
        ),
    ]

    summary = summarize(scores)

    assert (summary["cases"], summary["failures"]) == (3, 1)
    assert summary["search_recall"] == pytest.approx(0.3)
    assert summary["kept_precision"] == 0.5
    assert summary["llm_calls"] == 50


# --- the committed eval set ---------------------------------------------------------


def test_the_survey_cases_are_well_formed():
    cases = load_cases(EVAL_SET)

    assert len({c.id for c in cases}) == len(cases) >= 5
    for c in cases:
        date.fromisoformat(c.cutoff)
        assert c.gold == sorted(set(c.gold)), c.id
        assert c.survey not in c.gold, c.id  # a survey can't be its own answer
        assert 0 < len(c.gold) <= c.with_arxiv_id <= c.references, c.id
