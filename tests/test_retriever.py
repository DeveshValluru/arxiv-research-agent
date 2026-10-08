from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arxiv_agent.clients.arxiv import PaperSummary
from arxiv_agent.ingestion.chunker import chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.retrieval import retriever as retriever_module
from arxiv_agent.retrieval.retriever import (
    DenseRetriever,
    HybridRetriever,
    build_retriever,
)

FIXTURES = Path(__file__).parent / "fixtures"
PAPER = ("2499.00001", 1)
CHUNKS = chunk_paper(
    parse_arxiv_html((FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")),
    *PAPER,
)
# Against the query [1, 0, 0], dense ranks chunk 2 first and chunk 4 second.
VECTORS = np.array(
    [[0, 1, 0], [0.6, 0.8, 0], [1, 0, 0], [0, 0, 1], [0.8, 0.6, 0], [0, 0.6, 0.8]],
    dtype=np.float32,
)


class FakeEmbedder:
    model_id = "test/fake-embedder"

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1, 0, 0], dtype=np.float32)


class FakeReranker:
    # Puts the candidates in reverse order and records what it was given.
    model_id = "test/fake-reranker"

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def rerank(self, question, chunks, top_k=None):
        self.seen.append([chunk.chunk_id for chunk in chunks])
        ranked = list(reversed(chunks))
        return [(chunk, float(-i)) for i, chunk in enumerate(ranked)][:top_k]


def index(store):
    store.save_paper(
        PaperSummary(
            arxiv_id=PAPER[0],
            version=PAPER[1],
            title="Judging the Judges: A Tiny Test Paper",
            authors=["Ada Lovelace"],
            abstract="We study how language models grade other models.",
            published=datetime(2024, 11, 23, tzinfo=UTC),
            updated=datetime(2024, 11, 23, tzinfo=UTC),
            primary_category="cs.CL",
            categories=["cs.CL"],
        )
    )
    store.replace_chunks(*PAPER, CHUNKS)
    store.ensure_vector_table(FakeEmbedder.model_id, 3)
    store.save_vectors(FakeEmbedder.model_id, [c.chunk_id for c in CHUNKS], VECTORS)
    return store


@pytest.fixture
def indexed(store):
    return index(store)


def ids(hits) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def test_dense_retriever_is_plain_vector_search(indexed):
    retriever = DenseRetriever(indexed, FakeEmbedder())

    hits = retriever.retrieve("swapping order", PAPER, k=2)

    assert ids(hits) == [CHUNKS[2].chunk_id, CHUNKS[4].chunk_id]
    assert retriever.name == "dense(test/fake-embedder)"


def test_hybrid_puts_the_chunk_both_methods_like_first(indexed):
    # Dense ranks chunk 4 second; BM25 ranks it first for "swapping order".
    retriever = HybridRetriever(indexed, FakeEmbedder())

    hits = retriever.retrieve("swapping order", PAPER, k=2)

    assert ids(hits) == [CHUNKS[4].chunk_id, CHUNKS[2].chunk_id]
    assert retriever.name == "hybrid(test/fake-embedder+bm25)"


def test_reranker_orders_the_fused_candidates(indexed):
    reranker = FakeReranker()
    retriever = HybridRetriever(indexed, FakeEmbedder(), reranker=reranker)

    hits = retriever.retrieve("swapping order", PAPER, k=2)

    fused = ids(
        HybridRetriever(indexed, FakeEmbedder()).retrieve("swapping order", PAPER, k=20)
    )
    assert reranker.seen == [fused]
    assert ids(hits) == list(reversed(fused))[:2]
    assert retriever.name.endswith("+rerank(test/fake-reranker)")


def test_bm25_index_is_built_once_per_paper(indexed):
    retriever = HybridRetriever(indexed, FakeEmbedder())
    retriever.retrieve("swapping", PAPER, k=1)
    first = retriever._papers[PAPER]

    retriever.retrieve("judges", PAPER, k=1)

    assert retriever._papers[PAPER] is first


def test_unindexed_paper_returns_nothing(indexed):
    assert (
        HybridRetriever(indexed, FakeEmbedder()).retrieve("x", ("2499.99999", 1), 5)
        == []
    )


def test_build_retriever(indexed, monkeypatch):
    monkeypatch.setattr(retriever_module, "Reranker", FakeReranker)
    embedder = FakeEmbedder()

    assert isinstance(build_retriever("dense", indexed, embedder), DenseRetriever)
    assert build_retriever("hybrid", indexed, embedder).name == (
        "hybrid(test/fake-embedder+bm25)"
    )
    assert build_retriever("rerank", indexed, embedder).name.endswith(
        "+rerank(test/fake-reranker)"
    )


def test_a_paper_asked_about_before_it_was_indexed_works_once_it_is(store):
    # The API lets you ask, see "ingest it first", ingest, and ask again.
    store.ensure_vector_table(FakeEmbedder.model_id, 3)  # other papers' vectors
    retriever = HybridRetriever(store, FakeEmbedder())
    assert retriever.retrieve("swapping order", PAPER, k=2) == []

    index(store)

    assert ids(retriever.retrieve("swapping order", PAPER, k=2)) == [
        CHUNKS[4].chunk_id,
        CHUNKS[2].chunk_id,
    ]


def test_forget_drops_a_papers_cached_index(indexed):
    # After a re-ingest (new chunks, new ids) the cached index is stale.
    retriever = HybridRetriever(indexed, FakeEmbedder())
    retriever.retrieve("swapping", PAPER, k=1)

    retriever.forget(PAPER)
    DenseRetriever(indexed, FakeEmbedder()).forget(PAPER)  # nothing to forget

    assert PAPER not in retriever._papers
