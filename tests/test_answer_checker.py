from arxiv_agent.qa.checker import check_answer


def test_cited_answer_is_answered():
    result = check_answer(
        "Judges favor the first answer [S2]. Swapping helps [S1][S3].", 5
    )
    assert result.status == "answered"
    assert result.cited == [2, 1, 3]
    assert result.problems == []


def test_repeated_citation_is_listed_once():
    assert check_answer("A holds [S2]. B holds [S2].", 5).cited == [2]


def test_exact_refusal_is_refused():
    assert check_answer("The paper doesn't say.", 5).status == "refused"


def test_refusal_tolerates_whitespace_and_a_curly_apostrophe():
    assert check_answer("  The paper doesn’t say.\n", 5).status == "refused"


def test_refusal_plus_a_guess_is_invalid():
    result = check_answer("The paper doesn't say. Usually an A100 is used.", 5)
    assert result.status == "invalid"
    assert result.problems == ["no citations"]


def test_out_of_range_citations_are_invalid():
    result = check_answer("Position bias is common [S7]. Also [S0].", 5)
    assert result.status == "invalid"
    assert result.problems == [
        "cites S7, but only S1-S5 exist",
        "cites S0, but only S1-S5 exist",
    ]


def test_thinking_block_is_removed():
    result = check_answer(
        "<think>\nlet me check\n</think>\n\nJudges favor the first answer [S1].", 5
    )
    assert result.text == "Judges favor the first answer [S1]."
    assert result.status == "answered"
