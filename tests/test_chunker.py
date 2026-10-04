from pathlib import Path

import pytest

from arxiv_agent.ingestion.chunker import (
    _approx_tokens,
    _chunk_texts,
    _pack,
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


def test_pack_combines_small_units():
    assert _pack(["aaaa", "bbbb"], max_tokens=10, separator=" ") == ["aaaa bbbb"]


def test_pack_starts_new_chunk_when_full():
    a, b, c = "a" * 20, "b" * 20, "c" * 20  # 5 tokens each
    assert _pack([a, b, c], max_tokens=12, separator=" ") == [f"{a} {b}", c]


def test_pack_keeps_oversized_unit_alone():
    big = "x" * 100  # 25 tokens, bigger than max_tokens
    assert _pack(["aaaa", big, "bbbb"], max_tokens=10, separator=" ") == [
        "aaaa",
        big,
        "bbbb",
    ]


def test_pack_empty_input():
    assert _pack([], max_tokens=10, separator=" ") == []


def test_pack_never_loses_or_duplicates_text():
    units = [f"unit{n}" for n in range(50)]
    chunks = _pack(units, max_tokens=8, separator=" ")
    assert " ".join(chunks) == " ".join(units)


def test_chunk_texts_packs_small_paragraphs_together():
    assert _chunk_texts("aaaa\n\nbbbb\n\ncccc", max_tokens=10) == [
        "aaaa\n\nbbbb\n\ncccc"
    ]


def test_chunk_texts_splits_long_paragraph_and_keeps_order():
    long_paragraph = "First sentence here. Second sentence here. Third sentence here."
    text = f"Short intro.\n\n{long_paragraph}\n\nShort outro."

    assert _chunk_texts(text, max_tokens=12) == [
        "Short intro.",
        "First sentence here. Second sentence here.",
        "Third sentence here.",
        "Short outro.",
    ]


def test_chunk_texts_never_loses_words_and_respects_limit():
    long_paragraph = " ".join(f"Sentence number {n} is here." for n in range(20))
    text = f"Intro paragraph.\n\n{long_paragraph}\n\nOutro paragraph."

    chunks = _chunk_texts(text, max_tokens=20)

    assert " ".join(chunks).split() == text.split()
    assert all(_approx_tokens(chunk) <= 20 for chunk in chunks)
