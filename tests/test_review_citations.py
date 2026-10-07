import pytest

from arxiv_agent.review.citations import (
    code_check,
    finalize,
    parse_draft,
    render,
    split_sentences,
)
from arxiv_agent.review.reader import quote_problem
from arxiv_agent.review.state import Claim, Critique, ScreenedPaper, SentenceCheck


def claim(label: str, arxiv_id: str) -> Claim:
    return Claim(
        label=label,
        arxiv_id=arxiv_id,
        version=1,
        chunk_id=f"{arxiv_id}v1:0001",
        section="4 Results",
        claim="A claim.",
        quote="A quote from the paper.",
        passage="A passage. A quote from the paper.",
    )


CLAIMS = {
    "K1": claim("K1", "2406.07791"),
    "K2": claim("K2", "2406.07791"),  # same paper as K1
    "K3": claim("K3", "2305.17926"),
}


# --- splitting the draft --------------------------------------------------------


def test_split_keeps_abbreviations_and_decimals_inside_sentences():
    text = (
        "Judges such as GPT-4 (e.g. GPT-4 Turbo) agree 0.83 of the time [K1]. "
        "Zheng et al. report this too [K2]. Is it robust? Not always [K3]."
    )
    assert split_sentences(text) == [
        "Judges such as GPT-4 (e.g. GPT-4 Turbo) agree 0.83 of the time [K1].",
        "Zheng et al. report this too [K2].",
        "Is it robust?",
        "Not always [K3].",
    ]


def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before():
    assert split_sentences(
        "Judges prefer the first answer. [K1][K2] Order matters."
    ) == [
        "Judges prefer the first answer. [K1][K2]",
        "Order matters.",
    ]


def test_parse_draft_keeps_paragraphs_and_drops_headings():
    draft = (
        "## Related work\n\nFirst point [K1]. Second\npoint [K2, K3].\n\nThird [K3]."
    )

    sentences = parse_draft(draft)

    assert [(s.paragraph, s.text, s.labels) for s in sentences] == [
        (1, "First point [K1].", ["K1"]),
        (1, "Second point [K2, K3].", ["K2", "K3"]),
        (2, "Third [K3].", ["K3"]),
    ]


# --- code checks ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("sentence", "verdict"),
    [
        ("Judges prefer the first answer [K1].", None),
        ("Judges prefer the first answer.", "uncited"),
        ("Judges prefer the first answer [K7].", "bad_citation"),
        ("As arXiv:2306.05685 shows, judges agree [K1].", "bad_citation"),
        ("See https://arxiv.org/abs/2306.05685 [K1].", "bad_citation"),
        ("Judges were studied on arXiv in 2024 [K1].", None),  # the word is fine
    ],
)
def test_code_check(sentence, verdict):
    [parsed] = parse_draft(sentence)
    check = code_check(parsed, CLAIMS)
    assert (check.verdict if check else None) == verdict


# --- rendering ------------------------------------------------------------------


def test_render_turns_labels_into_one_citation_per_paper():
    assert render("Judges agree [K1][K2][K3].", CLAIMS) == (
        "Judges agree [arXiv:2406.07791, arXiv:2305.17926]."
    )
    assert (
        render("Judges agree [K2, K1].", CLAIMS) == "Judges agree [arXiv:2406.07791]."
    )


def test_finalize_keeps_passing_sentences_and_lists_cited_papers():
    def check(paragraph, sentence, labels, verdict):
        return SentenceCheck(
            paragraph=paragraph,
            sentence=sentence,
            labels=labels,
            verdict=verdict,
            reason="r",
        )

    kept = [
        ScreenedPaper(
            arxiv_id=arxiv_id,
            version=1,
            title=f"Paper {arxiv_id}",
            authors=[],
            published="2024-05-01",
            via="search",
            found_by=["q"],
            score=9,
            reason="r",
        )
        for arxiv_id in ("2406.07791", "2305.17926")
    ]
    checks = [
        check(0, "Order matters [K3].", ["K3"], "supported"),
        check(0, "All judges are biased [K1].", ["K1"], "overstated"),
        # Two claims from the same paper: one reference, not two.
        check(1, "Judges agree [K1][K2].", ["K1", "K2"], "unchecked"),
    ]

    result = finalize(
        {
            "claims": list(CLAIMS.values()),
            "kept": kept,
            "critique": Critique(verdict="give_up", checks=checks),
        }
    )

    assert result["review"] == (
        "Order matters [arXiv:2305.17926].\n\nJudges agree [arXiv:2406.07791]."
    )
    assert [p.arxiv_id for p in result["references"]] == ["2305.17926", "2406.07791"]
    assert result["removed"] == ["All judges are biased [K1]."]
    assert [[c.label for c in e.claims] for e in result["evidence"]] == [
        ["K3"],
        ["K1", "K2"],
    ]


# --- quote check ----------------------------------------------------------------

PASSAGE = (
    "We ran 5,000 comparisons. LLM judges prefer the answer shown first, "
    "even when it is wrong. Swapping the order reduces this bias."
)


@pytest.mark.parametrize(
    ("quote", "problem"),
    [
        ("LLM judges prefer the answer shown first, even when it is wrong.", None),
        ("llm judges prefer the answer shown first — even when it is wrong", None),
        ("LLM judges prefer the answer ... even when it is wrong.", None),
        ("LLM judges prefer the answer shown last.", "quote not found in the passage"),
        ("Swapping the order", "quote too short to check"),
        ("We ran 5,000 comparisons of LLM judges.", "quote not found in the passage"),
        ("judges prefer the answer shown firs", "quote not found in the passage"),
    ],
)
def test_quote_problem(quote, problem):
    assert quote_problem(quote, PASSAGE) == problem
