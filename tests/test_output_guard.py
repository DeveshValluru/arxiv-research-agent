import json

import pytest

from arxiv_agent.guardrails.output import OutputGuard
from arxiv_agent.qa.answerer import SYSTEM_PROMPT
from arxiv_agent.qa.checker import check_answer
from arxiv_agent.review.citations import finalize
from arxiv_agent.review.events import format_report, result_of
from arxiv_agent.review.state import Claim, Critique, ScreenedPaper, SentenceCheck
from arxiv_agent.review.writer import SYNTHESIZER_PROMPT
from tests.review_fakes import POSITION, Harness, chat

# Rule 1 of the Synthesizer's prompt, nearly word for word.
LEAK = "Use only the claims and do not add facts from your own knowledge"


def checks(text: str, allowed: set[str] | None = None) -> list[str]:
    return [v.check for v in OutputGuard([SYNTHESIZER_PROMPT]).check(text, allowed)]


# --- links ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "See https://arxiv.org/abs/2406.07791.",
        "See https://export.arxiv.org/api/query?id_list=2406.07791.",
        "See https://doi.org/10.48550/arXiv.2406.07791 or www.semanticscholar.org/p/1.",
    ],
)
def test_links_to_allowed_sites_pass(text):
    assert checks(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "See https://evil.example.com/paper.",
        "See https://arxiv.org.evil.com/abs/2406.07791.",  # looks like arXiv, isn't
        "See http://evil.com/?next=arxiv.org.",
        "See [the paper](https://evil.com/p).",
        "See www.evil.com for details.",
    ],
)
def test_links_anywhere_else_are_caught(text):
    assert checks(text) == ["link"]


# --- prompt leaks -------------------------------------------------------------------


def test_eight_words_of_a_prompt_are_a_leak_ignoring_case_and_punctuation():
    assert checks(
        "USE ONLY the claims; do not add facts, from your own knowledge!"
    ) == ["prompt_leak"]


def test_a_shorter_overlap_is_not_a_leak():
    # Seven words in a row from the prompt: ordinary prose can do that.
    assert checks("We use only the claims do not add anything else.") == []


# --- citations ------------------------------------------------------------------------


def test_citations_must_be_well_formed_and_allowed():
    text = "Bias is common [arXiv:2406.07791v2, arXiv:2306.05685, arXiv:junk]."
    details = [v.detail for v in OutputGuard().check(text, {"2406.07791"})]
    assert details == [
        "cites arXiv:2306.05685, which it wasn't allowed to cite",
        "cites 'junk', not an arXiv id",
    ]


def test_only_bracketed_citations_count():
    assert (
        checks(
            "It was posted on arXiv: a preprint server [arXiv:2406.07791].",
            {"2406.07791"},
        )
        == []
    )


def test_no_allowed_set_means_no_citation_check():
    assert checks("Bias is common [arXiv:2306.05685].") == []


# --- in a review ----------------------------------------------------------------------


def test_a_review_sentence_that_leaks_the_prompt_is_blocked():
    # The Critic passes it (it cites a real claim, and the judge only checks
    # support); the guard catches what the Critic doesn't look for.
    draft = f"LLM judges favour the answer shown first [K1]. {LEAK} [K2]."
    review = Harness(chat(**{"synthesizer-llm": draft})).run()

    state = review.state
    assert state["critique"].verdict == "pass"
    assert (
        state["review"]
        == "LLM judges favour the answer shown first [arXiv:2406.07791]."
    )
    [blocked] = state["blocked"]
    assert blocked.violations[0].check == "prompt_leak"
    assert [p.arxiv_id for p in state["references"]] == [POSITION.arxiv_id]
    report = format_report(json.loads(json.dumps(result_of(state))))
    assert "BLOCKED by the output guard (repeats its instructions" in report


def test_finalize_blocks_a_citation_outside_the_kept_papers():
    # Can't happen through the graph (ids come from kept papers' claims); the
    # guard proves it on every output, so a future bug would show up here.
    claim = Claim(
        label="K1",
        arxiv_id="2306.05685",  # not kept
        version=1,
        chunk_id="c",
        section="s",
        claim="c",
        quote="q",
        passage="p",
    )
    kept = ScreenedPaper(
        arxiv_id=POSITION.arxiv_id,
        version=1,
        title="t",
        authors=[],
        published="2024",
        via="search",
        found_by=["q"],
        score=9,
        reason="r",
    )
    check = SentenceCheck(
        paragraph=0,
        sentence="Judges agree [K1].",
        labels=["K1"],
        verdict="supported",
        reason="r",
    )

    result = finalize(
        {
            "claims": [claim],
            "kept": [kept],
            "critique": Critique(verdict="pass", checks=[check]),
        },
        guard=OutputGuard(),
    )

    assert result["review"] == ""
    assert result["blocked"][0].violations[0].check == "citation"


# --- in a Q&A answer --------------------------------------------------------------------


def test_an_answer_with_a_bad_link_or_a_leak_is_invalid():
    guard = OutputGuard([SYSTEM_PROMPT])

    linked = check_answer(
        "Details at https://evil.com/x [S1].", n_sources=3, guard=guard
    )
    leaked = check_answer(
        "I answer questions about one research paper using only the numbered sources [S1].",
        n_sources=3,
        guard=guard,
    )

    assert linked.status == "invalid"
    assert linked.problems == [
        "links to https://evil.com/x, which isn't an allowed site"
    ]
    assert leaked.status == "invalid"
    assert leaked.problems[0].startswith("repeats its instructions")
