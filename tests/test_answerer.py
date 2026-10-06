import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arxiv_agent.ingestion.chunker import CHUNKER_VERSION, chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.qa.answerer import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    Answerer,
    LLMUnavailableError,
    build_messages,
    format_sources,
)
from arxiv_agent.qa.checker import REFUSAL
from arxiv_agent.storage.chunk_store import SearchHit

FIXTURES = Path(__file__).parent / "fixtures"
CHUNKS = chunk_paper(
    parse_arxiv_html((FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")),
    "2499.00001",
    1,
)
HITS = [SearchHit(chunk=CHUNKS[4], score=0.8), SearchHit(chunk=CHUNKS[3], score=0.7)]
GOOD = "Swapping the order reduces the bias [S1]."


class FakeStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.searches: list[dict] = []

    def vector_search(self, model_id, query_vector, k=10, papers=None):
        self.searches.append({"model_id": model_id, "k": k, "papers": papers})
        return self.hits


class FakeEmbedder:
    model_id = "test/fake-embedder"

    def embed_query(self, query: str) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class FakeHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class FakeLLM:
    # Plays its outcomes in order (an answer string or an exception to raise);
    # the last one repeats forever.
    def __init__(self, *outcomes, finish_reason: str = "stop") -> None:
        self.outcomes = list(outcomes)
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def chat_completion(self, messages, model=None, max_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=outcome),
                    finish_reason=self.finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=300, completion_tokens=20, estimated_cost=0.0001
            ),
        )


class FakeObservation:
    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs
        self.updates: dict = {}
        self.trace_id = "abc123"

    def update(self, **kwargs) -> None:
        self.updates.update(kwargs)


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        observation = FakeObservation(kwargs)
        self.observations.append(observation)
        yield observation

    def named(self, name: str) -> list[FakeObservation]:
        return [o for o in self.observations if o.kwargs["name"] == name]


def make_answerer(*llms: FakeLLM, hits=HITS, langfuse=None, sleeps=None):
    providers = [f"fake-{letter}" for letter in "abc"[: len(llms)]]
    return Answerer(
        FakeStore(hits),
        FakeEmbedder(),
        model="test/fake-llm",
        providers=providers,
        clients=dict(zip(providers, llms)),
        langfuse=langfuse or FakeLangfuse(),
        sleep=(sleeps if sleeps is not None else []).append,
    )


def test_format_sources_labels_each_chunk_with_its_section_path():
    assert format_sources(HITS) == (
        "[S1] (2. Method > 2.1. Position Bias > 2.1.1. Swapping) "
        "Swapping the order reduces the bias by $\\Delta b$.\n\n"
        "[S2] (2. Method > 2.1. Position Bias) "
        "Judges prefer the first answer shown.\n\nPrompt Template Box Title"
    )


def test_build_messages_puts_rules_first_and_the_question_last():
    system, user = build_messages("Why swap?", HITS)

    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert user["content"].startswith("Sources:\n\n[S1]")
    assert user["content"].endswith("Question: Why swap?")


def test_prompt_and_checker_share_the_refusal_sentence():
    assert REFUSAL in SYSTEM_PROMPT


def test_ask_returns_the_full_record():
    result = make_answerer(FakeLLM(GOOD)).ask("Why swap?", "2499.00001", 1)

    assert result.answer.status == "answered"
    assert result.answer.cited == [1]
    assert [s.label for s in result.sources] == ["S1", "S2"]
    assert [s.chunk_id for s in result.sources] == [
        CHUNKS[4].chunk_id,
        CHUNKS[3].chunk_id,
    ]
    assert result.sources[0].section_path == CHUNKS[4].section_path
    assert (result.model, result.provider) == ("test/fake-llm", "fake-a")
    assert result.llm_attempts == 1
    assert (result.prompt_tokens, result.completion_tokens) == (300, 20)
    assert result.cost_usd == pytest.approx(0.0001)
    assert result.finish_reason == "stop"
    assert result.prompt_version == PROMPT_VERSION
    assert result.embedder == "test/fake-embedder"
    assert result.chunker_version == CHUNKER_VERSION
    assert result.trace_id == "abc123"
    assert result.retrieval_ms >= 0 and result.generation_ms >= 0


def test_ask_retrieves_k_chunks_from_this_paper_only():
    llm = FakeLLM(GOOD)
    answerer = make_answerer(llm)

    answerer.ask("Why swap?", "2499.00001", 1)

    assert answerer._store.searches == [
        {"model_id": "test/fake-embedder", "k": 5, "papers": [("2499.00001", 1)]}
    ]
    assert llm.calls[0]["model"] == "test/fake-llm"
    assert llm.calls[0]["messages"] == build_messages("Why swap?", HITS)


def test_refusal_comes_through():
    result = make_answerer(FakeLLM(REFUSAL)).ask("What GPU?", "2499.00001", 1)
    assert result.answer.status == "refused"


def test_cut_off_answer_is_invalid_even_if_its_citations_are_fine():
    llm = FakeLLM("Swapping helps [S1]. It also", finish_reason="length")

    result = make_answerer(llm).ask("Why swap?", "2499.00001", 1)

    assert result.answer.status == "invalid"
    assert result.answer.problems == [
        "finish_reason was 'length': the answer was cut off"
    ]


def test_unindexed_paper_fails_loudly():
    with pytest.raises(LookupError, match="ingest it first"):
        make_answerer(FakeLLM("unused"), hits=[]).ask("Why?", "2499.00001", 1)


def test_trace_is_ask_with_retrieve_and_llm_inside():
    langfuse = FakeLangfuse()

    make_answerer(FakeLLM(GOOD), langfuse=langfuse).ask("Why swap?", "2499.00001", 1)

    assert [(o.kwargs["name"], o.kwargs["as_type"]) for o in langfuse.observations] == [
        ("ask", "chain"),
        ("retrieve", "retriever"),
        ("llm", "generation"),
    ]
    [root], [retrieve], [llm] = (langfuse.named(n) for n in ("ask", "retrieve", "llm"))
    assert root.kwargs["input"] == {"question": "Why swap?", "paper": "2499.00001v1"}
    assert root.updates["output"] == GOOD
    assert root.updates["metadata"]["status"] == "answered"
    assert [s["chunk_id"] for s in retrieve.updates["output"]] == [
        CHUNKS[4].chunk_id,
        CHUNKS[3].chunk_id,
    ]
    assert llm.kwargs["model"] == "test/fake-llm"
    assert llm.updates["usage_details"] == {"input": 300, "output": 20}
    assert llm.updates["cost_details"] == {"total": pytest.approx(0.0001)}


def test_invalid_answer_marks_the_trace_as_a_warning():
    langfuse = FakeLangfuse()

    make_answerer(FakeLLM("No citations here."), langfuse=langfuse).ask(
        "Why swap?", "2499.00001", 1
    )

    [root] = langfuse.named("ask")
    assert root.updates["level"] == "WARNING"
    assert root.updates["status_message"] == "no citations"


def test_transient_error_is_retried_on_the_same_provider():
    langfuse, sleeps = FakeLangfuse(), []
    llm = FakeLLM(FakeHTTPError(429), GOOD)

    result = make_answerer(llm, langfuse=langfuse, sleeps=sleeps).ask(
        "Why swap?", "2499.00001", 1
    )

    assert (result.provider, result.llm_attempts) == ("fake-a", 2)
    assert len(sleeps) == 1 and 2 <= sleeps[0] <= 3
    first, second = langfuse.named("llm")
    assert first.updates["level"] == "ERROR"
    assert "level" not in second.updates


def test_fails_over_to_the_next_provider():
    sleeps = []
    busy, backup = FakeLLM(FakeHTTPError(429)), FakeLLM(GOOD)

    result = make_answerer(busy, backup, sleeps=sleeps).ask(
        "Why swap?", "2499.00001", 1
    )

    assert (result.provider, result.llm_attempts) == ("fake-b", 3)
    assert len(busy.calls) == 2 and len(backup.calls) == 1
    assert len(sleeps) == 1  # backoff between retries only, not before failover


def test_non_transient_error_is_not_retried():
    sleeps = []
    llm = FakeLLM(FakeHTTPError(400))

    with pytest.raises(FakeHTTPError):
        make_answerer(llm, FakeLLM(GOOD), sleeps=sleeps).ask("Why?", "2499.00001", 1)

    assert len(llm.calls) == 1 and sleeps == []


def test_every_provider_failing_raises():
    with pytest.raises(LLMUnavailableError, match="fake-a: FakeHTTPError"):
        make_answerer(FakeLLM(FakeHTTPError(503)), FakeLLM(FakeHTTPError(503))).ask(
            "Why?", "2499.00001", 1
        )


def test_tests_never_send_traces():
    # conftest.py turns tracing off, so the real client runs but sends nothing.
    assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"
    answerer = Answerer(
        FakeStore(HITS),
        FakeEmbedder(),
        model="test/fake-llm",
        providers=["fake-a"],
        clients={"fake-a": FakeLLM(GOOD)},
    )

    result = answerer.ask("Why swap?", "2499.00001", 1)

    assert result.answer.status == "answered"
    assert result.trace_id is None
