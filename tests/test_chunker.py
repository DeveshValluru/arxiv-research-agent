from pathlib import Path

import pytest

from arxiv_agent.ingestion.chunker import (
    _approx_tokens,
    _chunk_texts,
    _pack,
    _section_paths,
    _split_sentences,
    chunk_paper,
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


def test_pack_uses_the_given_token_counter():
    def count_words(text: str) -> int:
        return len(text.split())

    chunks = _pack(
        ["a b", "c d", "e f"], max_tokens=4, separator=" ", count_tokens=count_words
    )

    assert chunks == ["a b c d", "e f"]


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


def test_chunk_texts_splits_a_giant_sentence_by_words():
    giant = " ".join(f"word{n}" for n in range(100))

    chunks = _chunk_texts(giant, max_tokens=20)

    assert len(chunks) > 1
    assert all(_approx_tokens(chunk) <= 20 for chunk in chunks)
    assert " ".join(chunks).split() == giant.split()


CHUNKS = chunk_paper(PAPER, "2499.00001", 1)


def test_abstract_is_first_chunk():
    first = CHUNKS[0]
    assert first.kind == "abstract"
    assert first.index == 0
    assert first.text == "We study how language models grade other models."


def test_chunk_ids_are_numbered_and_stable():
    assert [c.chunk_id for c in CHUNKS[:3]] == [
        "2499.00001v1:0000",
        "2499.00001v1:0001",
        "2499.00001v1:0002",
    ]
    again = chunk_paper(PAPER, "2499.00001", 1)
    assert [c.chunk_id for c in again] == [c.chunk_id for c in CHUNKS]


def test_text_chunks_carry_their_section_path():
    swapping = next(c for c in CHUNKS if "Swapping the order" in c.text)
    assert swapping.kind == "text"
    assert swapping.section_path == [
        "2. Method",
        "2.1. Position Bias",
        "2.1.1. Swapping",
    ]


def test_embed_text_has_context_header_but_text_does_not():
    swapping = next(c for c in CHUNKS if "Swapping the order" in c.text)
    assert swapping.embed_text.startswith(
        "Judging the Judges: A Tiny Test Paper\n"
        "2. Method > 2.1. Position Bias > 2.1.1. Swapping\n\n"
    )
    assert not swapping.text.startswith("Judging the Judges")


def test_table_is_one_chunk_with_caption_and_markdown():
    [table] = [c for c in CHUNKS if c.kind == "table"]
    assert table.section_path == ["2. Method"]
    assert table.text.startswith("Table 1. Judge agreement with humans.")
    assert "| GPT-4 | 85% |" in table.text
    assert "Columns: | Judge | Agreement |" in table.embed_text
    assert "85%" not in table.embed_text


def test_empty_abstract_makes_no_chunk():
    no_abstract = PAPER.model_copy(update={"abstract": ""})
    chunks = chunk_paper(no_abstract, "2499.00001", 1)
    assert all(c.kind != "abstract" for c in chunks)
    assert all(c.text for c in chunks)


def test_no_text_is_lost():
    sections_text = " ".join(s.text for s in PAPER.sections)
    words_in = f"{PAPER.abstract} {sections_text}".split()
    words_out = " ".join(c.text for c in CHUNKS if c.kind != "table").split()
    assert words_out == words_in


REAL_PAGE = Path(__file__).parent.parent / "data" / "html" / "2411.15594v6.html"


@pytest.mark.skipif(not REAL_PAGE.exists(), reason="real page not downloaded")
def test_real_paper_chunks():
    paper = parse_arxiv_html(REAL_PAGE.read_text(encoding="utf-8"))
    chunks = chunk_paper(paper, "2411.15594", 6)

    sections_text = " ".join(s.text for s in paper.sections)
    words_in = f"{paper.abstract} {sections_text}".split()
    words_out = " ".join(c.text for c in chunks if c.kind != "table").split()

    assert words_out == words_in
    assert all(c.token_count <= 512 for c in chunks)
    assert [c.kind for c in chunks[:2]] == ["abstract", "abstract"]
    assert sum(c.kind == "table" for c in chunks) == 4
