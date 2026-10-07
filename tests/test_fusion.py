import pytest

from arxiv_agent.retrieval.fusion import reciprocal_rank_fusion

DENSE = ["A", "B", "C", "D"]
BM25 = ["D", "E", "A"]


def test_the_worked_example():
    fused = reciprocal_rank_fusion([DENSE, BM25])

    assert [item for item, _ in fused] == ["A", "D", "B", "E", "C"]
    scores = dict(fused)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["D"] == pytest.approx(1 / 64 + 1 / 61)
    assert scores["C"] == pytest.approx(1 / 63)


def test_agreement_beats_a_single_first_place():
    # Y is 5th in both lists, X is 1st in only one: 2/65 > 1/61.
    fused = reciprocal_rank_fusion(
        [["X", "a", "b", "c", "Y"], ["d", "e", "f", "g", "Y"]]
    )
    assert fused[0][0] == "Y"


def test_small_k_lets_a_single_first_place_win():
    # With k = 0, first place is worth 1.0 and fifth place 0.2: 1 > 2/5.
    fused = reciprocal_rank_fusion(
        [["X", "a", "b", "c", "Y"], ["d", "e", "f", "g", "Y"]], k=0
    )
    assert fused[0][0] == "X"


def test_one_ranking_keeps_its_order():
    assert [item for item, _ in reciprocal_rank_fusion([DENSE])] == DENSE


def test_ties_keep_the_order_items_were_first_seen():
    # B (2nd in the first list) and E (2nd in the second) score the same.
    fused = [item for item, _ in reciprocal_rank_fusion([["A", "B"], ["A", "E"]])]
    assert fused == ["A", "B", "E"]


def test_nothing_to_fuse():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
