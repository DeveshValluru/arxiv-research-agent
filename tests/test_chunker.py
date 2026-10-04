from pathlib import Path

import pytest

from arxiv_agent.ingestion.chunker import (
    _approx_tokens,
    _section_paths,
    _split_sentences,
)
from arxiv_agent.ingestion.html_parser import parse_arxiv_html

FIXTURES = Path(__file__).parent / "fixtures"
PAPER = parse_arxiv_html(
    (FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")
)


def test_approx_tokens_round_up():
    assert _approx_tokens("") == 0
    assert _approx_tokens("abcd") == 1
    assert _approx_tokens("abcde") == 2


def test_section_paths_follow_nesting():
    assert _section_paths(PAPER.sections) == [
        ["1. Introduction"],
        ["2. Method"],
        ["2. Method", "2.1. Position Bias"],
        ["2. Method", "2.1. Position Bias", "2.1.1. Swapping"],
    ]


def test_split_sentences_basic():
    assert _split_sentences("First sentence. Second one! Third? Fourth.") == [
        "First sentence.",
        "Second one!",
        "Third?",
        "Fourth.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Judges were studied by Zheng et al. (2023) in detail.",
        "Strong judges, e.g. GPT-4, agree with humans.",
        "As shown in Fig. 2 and Eq. 3, the gap is small.",
    ],
)
def test_split_sentences_respects_abbreviations(text):
    assert _split_sentences(text) == [text]


def test_split_sentences_keeps_decimals():
    assert _split_sentences("Agreement rose from 0.72 to 0.85. Bias fell.") == [
        "Agreement rose from 0.72 to 0.85.",
        "Bias fell.",
    ]
