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


class FakeLLM:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def chat_completion(self, messages, model=None, max_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                    finish_reason=self.finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=300, completion_tokens=20, estimated_cost=0.0001
            ),
        )


def make_answerer(llm: FakeLLM, hits: list[SearchHit] = HITS) -> Answerer:
    return Answerer(
        FakeStore(hits),
        FakeEmbedder(),
        model="test/fake-llm",
        provider="fake",
        client=llm,
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
    llm = FakeLLM("Swapping the order reduces the bias [S1].")

    result = make_answerer(llm).ask("Why swap?", "2499.00001", 1)

    assert result.answer.status == "answered"
    assert result.answer.cited == [1]
    assert [s.label for s in result.sources] == ["S1", "S2"]
    assert [s.chunk_id for s in result.sources] == [
        CHUNKS[4].chunk_id,
        CHUNKS[3].chunk_id,
    ]
    assert result.sources[0].section_path == CHUNKS[4].section_path
    assert (result.model, result.provider) == ("test/fake-llm", "fake")
    assert (result.prompt_tokens, result.completion_tokens) == (300, 20)
    assert result.cost_usd == pytest.approx(0.0001)
    assert result.finish_reason == "stop"
    assert result.prompt_version == PROMPT_VERSION
    assert result.embedder == "test/fake-embedder"
    assert result.chunker_version == CHUNKER_VERSION
    assert result.retrieval_ms >= 0 and result.generation_ms >= 0


def test_ask_retrieves_k_chunks_from_this_paper_only():
    llm = FakeLLM("Swapping helps [S1].")
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
