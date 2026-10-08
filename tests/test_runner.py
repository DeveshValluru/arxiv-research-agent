from pathlib import Path

import pytest

from arxiv_agent.evals.judge import JudgeError, Verdict
from arxiv_agent.evals.runner import (
    EvalItem,
    ItemScore,
    load_items,
    score_item,
    summarize,
)
from arxiv_agent.ingestion.chunker import chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.qa.answerer import QAResult, Repair, Source
from arxiv_agent.qa.checker import REFUSAL, CheckedAnswer
from arxiv_agent.qa.support import SentenceSupport

FIXTURES = Path(__file__).parent / "fixtures"
EVALS = Path(__file__).parent.parent / "evals"
CHUNKS = chunk_paper(
    parse_arxiv_html((FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")),
    "2499.00001",
    1,
)
ANSWER = "Swapping reduces bias [S1]."


class FakeJudge:
    def __init__(self, verdict: str = "correct", fail: bool = False) -> None:
        self.verdict = verdict
        self.fail = fail
        self.calls: list[dict] = []

    def grade(self, question, gold_answers, answer, false_premise=False):
        self.calls.append({"answer": answer, "false_premise": false_premise})
        if self.fail:
            raise JudgeError("unusable judge output: 'maybe'")
        return Verdict(verdict=self.verdict, reasoning="Matches the reference.")


def make_item(item_type: str = "answerable") -> EvalItem:
    unanswerable = item_type == "unanswerable"
    return EvalItem(
        id="survey-a01",
        source="survey",
        arxiv_id="2499.00001",
        version=1,
        question="Why swap?",
        type=item_type,
        answer_type=None if unanswerable else "abstractive",
        gold_answers=[] if unanswerable else ["Swapping reduces bias."],
        evidence=[] if unanswerable else ["Swapping the order reduces"],
        evidence_matched=not unanswerable,
    )


def make_result(text: str, status: str, source_index: int = 4) -> QAResult:
    chunk = CHUNKS[source_index]
    return QAResult(
        question="Why swap?",
        arxiv_id="2499.00001",
        version=1,
        answer=CheckedAnswer(
            text=text,
            status=status,
            cited=[] if status == "refused" else [1],
            problems=[],
        ),
        sources=[
            Source(
                label="S1",
                chunk_id=chunk.chunk_id,
                section_path=chunk.section_path,
                score=0.8,
                text=chunk.text,
            )
        ],
        model="test/fake-llm",
        provider="fake",
        llm_attempts=1,
        finish_reason="stop",
        prompt_tokens=300,
        completion_tokens=20,
        cost_usd=0.0001,
        retrieval_ms=40.0,
        generation_ms=1000.0,
        prompt_version=1,
        retriever="test/fake-retriever",
        chunker_version=3,
        trace_id="abc123",
    )


def test_answerable_answer_is_graded_by_the_judge():
    judge = FakeJudge("partially_correct")

    score = score_item(make_item(), make_result(ANSWER, "answered"), CHUNKS, judge)

    assert (score.correctness, score.graded_by) == (0.5, "judge")
    assert score.judge_reasoning == "Matches the reference."
    assert judge.calls == [{"answer": ANSWER, "false_premise": False}]
    assert score.f1 == pytest.approx(1.0)
    assert score.refusal_ok is True
    assert score.evidence_hit is True
    assert (score.cost_usd, score.latency_ms, score.trace_id) == (
        0.0001,
        1040.0,
        "abc123",
    )


def test_evidence_miss_when_the_gold_chunk_was_not_retrieved():
    result = make_result(ANSWER, "answered", source_index=0)
    score = score_item(make_item(), result, CHUNKS, FakeJudge())
    assert score.evidence_hit is False


@pytest.mark.parametrize(
    ("item_type", "status", "expected"),
    [
        ("unanswerable", "refused", 1.0),
        ("unanswerable", "answered", 0.0),
        ("answerable", "refused", 0.0),
        ("false_premise", "refused", 0.5),
    ],
)
def test_certain_cases_are_graded_by_rule_without_the_judge(
    item_type, status, expected
):
    judge = FakeJudge()
    text = REFUSAL if status == "refused" else ANSWER

    score = score_item(make_item(item_type), make_result(text, status), CHUNKS, judge)

    assert (score.correctness, score.graded_by) == (expected, "rule")
    assert judge.calls == []


def test_false_premise_answer_goes_to_the_judge_with_its_rule():
    judge = FakeJudge("incorrect")
    result = make_result(ANSWER, "answered")

    score = score_item(make_item("false_premise"), result, CHUNKS, judge)

    assert (score.correctness, score.graded_by) == (0.0, "judge")
    assert judge.calls[0]["false_premise"] is True
    assert score.f1 is None and score.refusal_ok is None


def test_judge_failure_is_recorded_not_counted_as_wrong():
    result = make_result(ANSWER, "answered")

    score = score_item(make_item(), result, CHUNKS, FakeJudge(fail=True))

    assert score.correctness is None
    assert score.error.startswith("judge: unusable judge output")


def test_summarize():
    common = {"cost_usd": 0.001}
    scores = [
        ItemScore(
            id="a1",
            source="survey",
            type="answerable",
            status="answered",
            correctness=1.0,
            f1=0.8,
            refusal_ok=True,
            evidence_hit=True,
            latency_ms=1000.0,
            **common,
        ),
        ItemScore(
            id="a2",
            source="qasper",
            type="answerable",
            status="refused",
            correctness=0.0,
            f1=0.0,
            refusal_ok=False,
            evidence_hit=None,
            latency_ms=3000.0,
            **common,
        ),
        ItemScore(
            id="u1",
            source="qasper",
            type="unanswerable",
            status="refused",
            correctness=1.0,
            refusal_ok=True,
            latency_ms=2000.0,
            **common,
        ),
        ItemScore(
            id="a3",
            source="survey",
            type="answerable",
            status="answered",
            correctness=None,
            error="judge: bad",
            f1=0.5,
            refusal_ok=True,
            evidence_hit=False,
            latency_ms=4000.0,
            **common,
        ),
        ItemScore(id="a4", source="survey", type="answerable", error="LLM down"),
    ]

    summary = summarize(scores)

    assert summary["questions"] == 5
    assert (summary["run_failures"], summary["judge_failures"]) == (1, 1)
    assert summary["correctness"] == pytest.approx(2 / 3)
    assert summary["correctness_by_type"] == {
        "answerable": 0.5,
        "unanswerable": 1.0,
        "false_premise": None,
    }
    assert summary["correctness_by_source"] == {"qasper": 0.5, "survey": 1.0}
    assert summary["refusal_accuracy"] == 1.0
    assert summary["false_refusal_rate"] == pytest.approx(1 / 3)
    assert summary["evidence_recall"] == 0.5
    assert summary["evidence_unknown"] == 1
    assert summary["token_f1"] == pytest.approx(1.3 / 3)
    assert summary["cost_usd"] == pytest.approx(0.004)
    assert (summary["latency_ms_p50"], summary["latency_ms_p95"]) == (2000.0, 4000.0)


def test_both_eval_files_load_into_the_schema():
    items = load_items([EVALS / "qa_qasper.jsonl", EVALS / "qa_survey.jsonl"])
    assert len(items) == 50
    assert len({item.id for item in items}) == 50


def test_score_records_what_the_support_check_did():
    result = make_result(REFUSAL, "refused").model_copy(
        update={
            "support_ms": 1500.0,
            "support": [
                SentenceSupport(
                    sentence="All judges agree [S1].",
                    sources=[1],
                    verdict="overstated",
                    reason="One judge was tested.",
                )
            ],
        }
    )

    score = score_item(make_item("answerable"), result, CHUNKS, FakeJudge())

    assert (score.flagged_sentences, score.support_refused) == (1, True)
    assert score.latency_ms == 40.0 + 1000.0 + 1500.0  # the check counts as latency
    summary = summarize([score])
    assert (summary["support_flagged_rate"], summary["support_refusals"]) == (1.0, 1)


def test_score_records_a_repair_only_when_its_rewrite_was_used():
    flagged = SentenceSupport(
        sentence="All judges agree [S1].",
        sources=[1],
        verdict="overstated",
        reason="One judge was tested.",
    )
    results = [
        make_result(ANSWER, "answered").model_copy(
            update={
                "support": [flagged],
                "repair": Repair(outcome=outcome, text=ANSWER, support=[]),
            }
        )
        for outcome in ("repaired", "fallback")
    ]

    scores = [
        score_item(make_item("answerable"), r, CHUNKS, FakeJudge()) for r in results
    ]

    assert [s.repaired for s in scores] == [True, False]
    assert summarize(scores)["support_repaired"] == 1
