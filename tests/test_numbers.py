import pytest

from arxiv_agent.review.citations import code_check, parse_draft
from arxiv_agent.review.numbers import missing_numbers, result_numbers
from arxiv_agent.review.state import Claim

PASSAGE = "Judges prefer the first answer in 62.0% of 12,000 pairs (Table 3)."


def numbers(text: str) -> list[str]:
    return [m.group(1) for m in result_numbers(text)]


def test_result_numbers_skip_names_years_references_and_small_counts():
    text = (
        "GPT-4 and Qwen3-32B reach 81% agreement on 2,500 items in 2024 "
        "(see Table 12), after 3 epochs, with p < 0.05."
    )
    assert numbers(text) == ["81", "2,500", "0.05"]


@pytest.mark.parametrize(
    ("sentence", "missing"),
    [
        ("Judges prefer the first answer in 62% of cases.", []),  # 62 == 62.0
        ("They compare 12000 pairs.", []),  # 12000 == 12,000
        ("Judges prefer the first answer in 69% of cases.", ["69"]),
        ("Judges prefer the first answer in 62.5% of 12,000 pairs.", ["62.5"]),
    ],
)
def test_missing_numbers_compares_values(sentence, missing):
    assert missing_numbers(sentence, [PASSAGE]) == missing


def test_the_critic_rejects_a_number_its_evidence_doesnt_contain():
    claims = {
        "K1": Claim(
            label="K1",
            arxiv_id="2406.07791",
            version=1,
            chunk_id="c1",
            section="Results",
            claim="c",
            quote="q",
            passage=PASSAGE,
        )
    }
    [right] = parse_draft("Judges prefer the first answer in 62% of cases [K1].")
    [wrong] = parse_draft("Judges prefer the first answer in 98% of cases [K1].")

    assert code_check(right, claims) is None
    check = code_check(wrong, claims)
    assert (check.verdict, check.reason) == (
        "unsupported",
        "98 isn't in the passages it cites",
    )
