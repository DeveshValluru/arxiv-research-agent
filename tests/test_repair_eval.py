import pytest

from arxiv_agent.evals.repair_eval import (
    Outcome,
    RepairCase,
    RepairResult,
    break_answer,
    break_gone,
    others_kept,
    summarize,
)
from arxiv_agent.qa.support import SentenceSupport

INTRO = "The paper uses two datasets."
RESULT = "The model reaches 85.3% accuracy on SQuAD [S2]."
ANSWER = f"{INTRO} {RESULT}"
SUPPORT = [
    SentenceSupport(sentence=INTRO, sources=[], verdict="uncited", reason=""),
    SentenceSupport(sentence=RESULT, sources=[2], verdict="supported", reason=""),
]


def case(kind: str = "number_changed", marker: str | None = "85.1% accuracy"):
    return RepairCase(
        id=f"q1:{kind}",
        item_id="q1",
        kind=kind,
        question="How accurate is it?",
        gold_answers=["85.3%"],
        arxiv_id="2499.00001",
        version=1,
        chunk_ids=["c1", "c2"],
        original=ANSWER,
        broken=ANSWER,
        sentence=RESULT if kind != "real" else "",
        broken_sentence="",
        marker=marker,
    )


def test_each_kind_breaks_the_first_supported_sentence_and_marks_what_it_added():
    made = {kind: rest for kind, *rest in break_answer(ANSWER, SUPPORT)}

    assert made["number_changed"] == [
        f"{INTRO} The model reaches 85.1% accuracy on SQuAD [S2].",
        RESULT,
        "The model reaches 85.1% accuracy on SQuAD [S2].",
        "85.1% accuracy",
    ]
    assert made["negated"][3] == "It is not the case that"
    assert made["overgeneralized"][3] == "Across all models, datasets and settings"
    assert all(broken.startswith(INTRO) for broken, *_ in made.values())


def test_a_break_whose_marker_is_already_in_the_answer_is_skipped():
    answer = f"It is not the case that costs drop. {RESULT}"
    support = [SUPPORT[1].model_copy(update={"sentence": RESULT})]

    kinds = [kind for kind, *_ in break_answer(answer, support)]

    assert kinds == ["number_changed", "overgeneralized"]


def test_scores_compare_the_final_text_with_the_break_and_the_original():
    broken = case()

    assert break_gone(broken, f"{INTRO} {RESULT}") is True
    assert break_gone(broken, "It reaches 85.1% ACCURACY [S2].") is False
    assert others_kept(broken, RESULT) == 0.0  # the intro was dropped
    assert others_kept(broken, f"{INTRO} Rewritten [S2].") == 1.0
    real = case("real", None)
    assert (break_gone(real, ""), others_kept(real, "")) == (None, None)


def result(kind: str, flagged: bool, remove: float, repair: float) -> RepairResult:
    def outcome(score: float, gone: bool) -> Outcome:
        status = "refused" if score == 0 else "answered"
        return Outcome(
            text="", status=status, break_gone=gone, others_kept=1.0, correctness=score
        )

    return RepairResult(
        id=kind,
        kind=kind,
        flagged=flagged,
        flagged_break=None if kind == "real" else flagged,
        false_flags=0,
        original=1.0,
        actions={
            "none": outcome(1.0, False),
            "remove": outcome(remove, flagged),
            "repair": outcome(repair, flagged),
        },
        repair_outcome="repaired" if flagged else None,
        repair_ms=3000.0 if flagged else None,
    )


def test_summary_splits_broken_and_real_cases_and_where_the_check_fired():
    results = [
        result("negated", True, 0.0, 1.0),
        result("overgeneralized", False, 1.0, 1.0),
        result("real", True, 0.0, 1.0),
    ]

    s = summarize(results)

    broken = s["broken"]
    assert (broken["cases"], broken["flagged_break"]) == (2, 0.5)
    assert broken["all"]["remove"]["correctness"] == 0.5
    assert broken["flagged"]["repair"]["correctness"] == 1.0
    assert broken["flagged"]["remove"]["refused"] == 1.0
    assert broken["by_kind"]["negated"]["repair"]["break_gone"] == 1.0
    assert s["real"]["all"]["repair"]["correctness"] == 1.0
    assert s["repair"] == {"used": 2, "fallbacks": 0, "ms_p50": pytest.approx(3000)}
