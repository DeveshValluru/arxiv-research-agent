import threading
import time

import pytest

from arxiv_agent.evals.qa_run import eval_sets, run_questions
from arxiv_agent.llm import LLMUnavailableError
from tests.test_answerer import FakeLangfuse
from tests.test_runner import ANSWER, CHUNKS, FakeJudge, make_item, make_result


class FakeAnswerer:
    # Slow enough that concurrent runs overlap; fails for one question.
    def __init__(self) -> None:
        self.active = 0
        self.most_active = 0
        self._lock = threading.Lock()

    def ask(self, question, arxiv_id, version):
        with self._lock:
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        if question == "down?":
            raise LLMUnavailableError("every provider failed")
        if question == "bug?":
            raise IndexError("string index out of range")
        return make_result(ANSWER, "answered")


class FakeStore:
    def get_chunks(self, arxiv_id, version):
        return CHUNKS


def test_questions_run_repeatedly_and_concurrently_and_come_back_in_order():
    items = [
        make_item().model_copy(update={"id": "a"}),
        make_item().model_copy(update={"id": "b", "question": "down?"}),
    ]
    answerer = FakeAnswerer()

    scores = run_questions(
        items,
        answerer,
        FakeJudge(),
        FakeStore(),
        run_id="run",
        langfuse=FakeLangfuse(),
        repeats=2,
        concurrency=4,
    )

    assert [(s.id, s.repeat) for s in scores] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert [s.correctness for s in scores] == [1.0, 1.0, None, None]
    assert scores[2].error.startswith("LLMUnavailableError")
    assert answerer.most_active > 1


def test_every_qa_eval_set_is_found_including_new_ones(tmp_path):
    for name in ("qa_qasper.jsonl", "qa_flagged.jsonl", "retrieval_x.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert [p.name for p in eval_sets(tmp_path)] == [
        "qa_flagged.jsonl",
        "qa_qasper.jsonl",
    ]


def test_a_bug_is_not_mistaken_for_a_provider_failure():
    # IndexError is a LookupError: catching LookupError hid a real race (7.2).
    items = [make_item().model_copy(update={"question": "bug?"})]

    with pytest.raises(IndexError):
        run_questions(
            items,
            FakeAnswerer(),
            FakeJudge(),
            FakeStore(),
            run_id="run",
            langfuse=FakeLangfuse(),
            concurrency=2,
        )
