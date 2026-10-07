from pathlib import Path

import pytest

from arxiv_agent.evals.qa_scoring import (
    answer_tokens,
    best_f1,
    evidence_hit,
    gold_chunk_ids,
    refusal_correct,
    token_f1,
)
from arxiv_agent.ingestion.chunker import chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html

FIXTURES = Path(__file__).parent / "fixtures"
CHUNKS = chunk_paper(
    parse_arxiv_html((FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")),
    "2499.00001",
    1,
)


def test_answer_tokens_drop_case_punctuation_articles_and_citations():
    assert answer_tokens("The model fine-tunes LLaMA-7B [S2].") == [
        "model",
        "fine",
        "tunes",
        "llama",
        "7b",
    ]


def test_identical_answers_score_one():
    assert token_f1("LLaMA-7B", "LLaMA-7B") == 1.0


def test_citations_do_not_cost_points():
    assert token_f1("LLaMA-7B [S2].", "LLaMA-7B") == 1.0


def test_extra_words_lower_precision():
    # predicted: llama 7b model (3), gold: llama 7b (2), shared 2
    # precision 2/3, recall 1 -> F1 = 0.8
    assert token_f1("The LLaMA-7B model.", "LLaMA-7B") == pytest.approx(0.8)


def test_no_shared_words_score_zero():
    assert token_f1("Vicuna", "LLaMA-7B") == 0.0


def test_f1_cannot_see_negation():
    # A known weakness, and the reason the LLM judge is the main score.
    assert token_f1("It does not help.", "It does help.") > 0.8


def test_best_f1_uses_the_closest_gold_answer():
    golds = ["Chatbot Arena", "MLLM-as-a-Judge"]
    assert best_f1("MLLM-as-a-Judge [S1].", golds) == 1.0
    assert best_f1("anything", []) == 0.0


@pytest.mark.parametrize(
    ("item_type", "status", "expected"),
    [
        ("unanswerable", "refused", True),
        ("unanswerable", "answered", False),
        ("unanswerable", "invalid", False),
        ("answerable", "answered", True),
        ("answerable", "invalid", True),
        ("answerable", "refused", False),
        ("false_premise", "refused", None),
    ],
)
def test_refusal_correct(item_type, status, expected):
    assert refusal_correct(item_type, status) is expected


def test_gold_chunk_ids_matches_short_phrases_exactly():
    assert gold_chunk_ids(CHUNKS, ["Swapping the order reduces"]) == {
        CHUNKS[4].chunk_id
    }


def test_gold_chunk_ids_matches_long_evidence_sentence_by_sentence():
    # A whole paragraph never appears verbatim, but its second sentence does.
    evidence = [
        (
            "This opening sentence is nowhere in the paper at all. "
            "We study how language models grade other models."
        )
    ]
    assert gold_chunk_ids(CHUNKS, evidence) == {CHUNKS[0].chunk_id}


def test_gold_chunk_ids_without_evidence_is_empty():
    assert gold_chunk_ids(CHUNKS, []) == set()


def test_evidence_hit():
    gold = {"p:0004"}
    assert evidence_hit(gold, ["p:0001", "p:0004"]) is True
    assert evidence_hit(gold, ["p:0001", "p:0002"]) is False
    assert evidence_hit(set(), ["p:0001"]) is None
