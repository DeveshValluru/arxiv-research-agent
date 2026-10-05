import pytest

from arxiv_agent.evals.retrieval import (
    first_hit_rank,
    gold_indices,
    mean_reciprocal_rank,
    normalize,
    recall_at_k,
)

RANKS = [1, 3, None, 2, 7]


def test_normalize_handles_case_quotes_and_spacing():
    assert normalize("Cohen’s   Kappa\n") == "cohen's kappa"


def test_gold_indices_matches_any_evidence_phrase():
    chunks = [
        "Judges prefer the FIRST answer.",
        "Unrelated text.",
        "We use Cohen’s Kappa here.",
    ]
    assert gold_indices(chunks, ["first answer", "cohen's kappa"]) == {0, 2}


def test_first_hit_rank_is_one_based():
    assert first_hit_rank([4, 2, 9], gold={2}) == 2
    assert first_hit_rank([4, 2, 9], gold={4}) == 1
    assert first_hit_rank([4, 2, 9], gold={5}) is None


def test_recall_at_k():
    assert recall_at_k(RANKS, 1) == pytest.approx(0.2)
    assert recall_at_k(RANKS, 3) == pytest.approx(0.6)
    assert recall_at_k(RANKS, 5) == pytest.approx(0.6)


def test_mean_reciprocal_rank():
    expected = (1 + 1 / 3 + 0 + 1 / 2 + 1 / 7) / 5
    assert mean_reciprocal_rank(RANKS) == pytest.approx(expected)
