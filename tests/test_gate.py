import pytest

from arxiv_agent.evals.gate import (
    FLOOR,
    compare,
    make_baseline,
    pooled_variance,
    render_markdown,
    stratified_subset,
)
from arxiv_agent.evals.runner import EvalItem, ItemScore

META = {"judge_model": "judge/x", "judge_prompt_version": 2}


def item(id: str, source: str = "qasper", type: str = "answerable") -> EvalItem:
    return EvalItem(
        id=id,
        source=source,
        arxiv_id="2499.00001",
        version=1,
        question="?",
        type=type,
        answer_type=None,
        gold_answers=[],
        evidence=[],
        evidence_matched=False,
    )


def runs(scores: dict[str, list[float | None]]) -> list[ItemScore]:
    return [
        ItemScore(
            id=item_id,
            source="qasper",
            type="answerable",
            repeat=r,
            status=None if value is None else "answered",
            correctness=value,
            trace_id=f"{item_id}-{r}",
        )
        for item_id, values in scores.items()
        for r, value in enumerate(values)
    ]


# 20 questions, 3 baseline runs: 16 always right, 4 that flip once.
BASE = {f"q{i}": [1.0, 1.0, 1.0] for i in range(16)} | {
    f"q{i}": [1.0, 0.0, 1.0] for i in range(16, 20)
}


def test_the_subset_keeps_every_group_in_proportion_and_doesnt_reshuffle():
    items = [item(f"a{i}") for i in range(19)]
    items += [item(f"u{i}", type="unanswerable") for i in range(8)]
    items += [item(f"s{i}", source="survey") for i in range(3)]

    subset = stratified_subset(items, 10)

    groups = [(i.source, i.type) for i in subset]
    assert len(subset) == 10
    assert groups.count(("qasper", "answerable")) == 6
    assert groups.count(("qasper", "unanswerable")) == 3
    assert groups.count(("survey", "answerable")) == 1
    # Adding a question to one group doesn't change who is picked in a group
    # whose quota stays the same.
    more = stratified_subset([*items, item("u99", type="unanswerable")], 10)

    def qasper_answerable(chosen):
        return {i.id for i in chosen if i.id.startswith("a")}

    assert qasper_answerable(more) == qasper_answerable(subset)


def test_pooled_variance_weights_each_question_by_its_runs():
    baseline = make_baseline(runs(BASE), META)

    assert baseline.items["q0"].var == 0.0
    assert baseline.items["q16"].var == pytest.approx(1 / 3)
    assert pooled_variance(baseline.items) == pytest.approx(4 / 20 * (1 / 3))


def test_the_same_system_passes():
    baseline = make_baseline(runs(BASE), META)
    candidate = {i: v[:2] for i, v in BASE.items()}

    result = compare(baseline, runs(candidate), META)

    assert result.status == "pass"
    assert result.questions == 20
    assert result.tolerance >= FLOOR


def test_a_drop_larger_than_noise_fails_and_names_the_questions():
    baseline = make_baseline(runs(BASE), META)
    candidate = {i: v[:2] for i, v in BASE.items()} | {
        "q0": [0.0, 0.0],
        "q1": [0.0, 0.0],
        "q2": [0.0, 0.5],
    }

    result = compare(baseline, runs(candidate), META)

    assert result.status == "fail"
    assert result.delta < -result.tolerance
    assert [c.id for c in result.worse] == ["q0", "q1", "q2"]
    assert result.worse[2].trace_id == "q2-0"  # its worst run


def test_one_question_flipping_is_within_noise():
    baseline = make_baseline(runs(BASE), META)
    candidate = {i: v[:2] for i, v in BASE.items()} | {"q0": [0.0, 1.0]}

    assert compare(baseline, runs(candidate), META).status == "pass"


def test_a_different_judge_cant_be_compared():
    baseline = make_baseline(runs(BASE), META)

    result = compare(baseline, runs(BASE), META | {"judge_prompt_version": 3})

    assert result.status == "fail"
    assert "update the baseline" in result.problems[0]


def test_too_many_unscored_runs_is_inconclusive():
    baseline = make_baseline(runs(BASE), META)
    candidate = {i: [None, 1.0] for i in BASE}

    result = compare(baseline, runs(candidate), META)

    assert result.status == "inconclusive"
    assert "rerun" in result.problems[0]


def test_new_questions_are_reported_not_gated():
    baseline = make_baseline(runs(BASE), META)

    result = compare(baseline, runs(BASE | {"new": [0.0, 0.0]}), META)

    assert (result.status, result.new) == ("pass", ["new"])


def test_the_report_shows_the_verdict_and_links_traces():
    baseline = make_baseline(runs(BASE), META)
    candidate = BASE | {"q0": [0.0, 0.0], "q1": [0.0, 0.0], "q2": [0.0, 0.0]}
    result = compare(baseline, runs(candidate), META)

    report = render_markdown(
        result, META | {"repeats": 3, "suite": "pr"}, lambda t: f"https://lf/{t}"
    )

    assert report.startswith("## ❌ Q&A eval gate: fail")
    assert "- `q0`: 1.00 → 0.00 ([trace](https://lf/q0-0))" in report


def test_the_console_report_has_no_characters_a_windows_console_cant_print():
    baseline = make_baseline(runs(BASE), META)
    candidate = BASE | {"q0": [0.0, 0.0], "q1": [0.0, 0.0], "q2": [0.0, 0.0]}
    result = compare(baseline, runs(candidate), META)

    report = render_markdown(result, META | {"repeats": 3, "suite": "pr"}, plain=True)

    report.encode("cp1252")  # what a redirected Windows console uses
    assert report.startswith("## Q&A eval gate: fail")
