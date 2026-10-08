import asyncio

import pytest

from arxiv_agent.evals.judge_eval import (
    Base,
    JudgedCase,
    build_cases,
    candidate_sentences,
    change_number,
    judge_case,
    negate,
    overgeneralize,
    summarize,
)
from tests.review_fakes import FakeChat

FINDING = "Judges prefer the first answer in 62% of the cases (Zheng et al., 2023)."


@pytest.mark.parametrize(
    ("sentence", "changed"),
    [
        (
            FINDING,
            "Judges prefer the first answer in 49% of the cases (Zheng et al., 2023).",
        ),
        (
            "As shown in Table 3, GPT-4 reaches an agreement of 0.81 with experts.",
            "As shown in Table 3, GPT-4 reaches an agreement of 0.98 with experts.",
        ),
        (
            "We collect 12,000 pairs from Qwen3-32B outputs.",
            "We collect 24,000 pairs from Qwen3-32B outputs.",
        ),
    ],
)
def test_change_number_alters_a_finding_not_a_name_year_or_reference(sentence, changed):
    assert change_number(sentence) == changed


def test_small_counts_and_footnote_markers_are_not_findings():
    assert change_number("We train for 3 epochs on 2 GPUs (Table 12).") is None


def test_negate_and_overgeneralize():
    assert negate("The judge is biased toward long answers.") == (
        "The judge is not biased toward long answers."
    )
    assert negate("Judges prefer long answers.") == (
        "It is not the case that judges prefer long answers."
    )
    assert overgeneralize("GPT-4 prefers long answers.") == (
        "Across all models, datasets and settings, gPT-4 prefers long answers."
    )


def test_candidate_sentences_are_results_in_plain_prose():
    text = (
        f"{FINDING} Twitter has more than 500 million tweets per day around the world.\n"
        "Our model achieves 91.2% accuracy, see https://example.com for the code and data."
    )
    assert candidate_sentences(text) == [FINDING]


def base(arxiv_id: str, sentence: str) -> Base:
    return Base(
        arxiv_id=arxiv_id,
        chunk_id=f"{arxiv_id}v1:0001",
        section="Results",
        sentence=sentence,
        passage=f"Some context. {sentence}",
    )


def test_each_sentence_gives_five_cases_and_the_wrong_passage_is_another_papers():
    cases = build_cases([base("2499.00001", FINDING), base("2499.00002", FINDING)])

    first = [c for c in cases if c.id.startswith("2499.00001")]
    assert [c.kind for c in first] == [
        "supported",
        "number_changed",
        "negated",
        "overgeneralized",
        "wrong_passage",
    ]
    assert [c.supported for c in first] == [True, False, False, False, False]
    assert first[-1].arxiv_id == "2499.00002"


def test_summary_separates_catches_from_false_alarms():
    def judged(kind, supported, verdict):
        return JudgedCase(
            id=kind, kind=kind, supported=supported, verdict=verdict, reason="r"
        )

    summary = summarize(
        [
            judged("supported", True, "supported"),
            judged("supported", True, "overstated"),  # a false alarm
            judged("negated", False, "unsupported"),  # caught
            judged("number_changed", False, "supported"),  # missed
            judged("overgeneralized", False, "error"),  # not scored
        ]
    )

    assert (summary["caught"], summary["false_alarms"], summary["errors"]) == (
        0.5,
        0.5,
        1,
    )
    assert summary["by_kind"]["negated"] == {"cases": 1, "correct": 1.0}


def test_judge_case_sends_the_production_prompt_and_reads_the_verdict():
    [case] = [
        c
        for c in build_cases([base("2499.00001", FINDING), base("2499.00002", FINDING)])
        if c.kind == "negated" and c.id.startswith("2499.00001")
    ]
    judge = FakeChat({"judge-eval": '{"verdict": "unsupported", "reason": "Negated."}'})

    result = asyncio.run(judge_case(judge, case, asyncio.Semaphore(1)))

    assert (result.verdict, result.supported) == ("unsupported", False)
    [[system, user]] = judge.prompts("judge-eval")
    assert "You check one sentence" in system["content"]
    assert user["content"].startswith(f"Sentence: {case.sentence[:-1]} [K1].")


def test_an_unreadable_verdict_is_an_error_not_a_score():
    case = build_cases([base("2499.00001", FINDING), base("2499.00002", FINDING)])[0]
    judge = FakeChat({"judge-eval": "Looks fine to me."})

    assert asyncio.run(judge_case(judge, case, asyncio.Semaphore(1))).verdict == "error"
