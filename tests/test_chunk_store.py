from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arxiv_agent.clients.arxiv import PaperSummary
from arxiv_agent.ingestion.chunker import CHUNKER_VERSION, chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.storage.chunk_store import vector_table

FIXTURES = Path(__file__).parent / "fixtures"
PARSED = parse_arxiv_html(
    (FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")
)
CHUNKS = chunk_paper(PARSED, "2499.00001", 1)
MODEL = "test/fake-embedder"
# One 3-d unit vector per fixture chunk; against the query [1, 0, 0] the
# inner products are 0.0, 0.6, 1.0, 0.0, 0.8, 0.0
VECTORS = np.array(
    [[0, 1, 0], [0.6, 0.8, 0], [1, 0, 0], [0, 0, 1], [0.8, 0.6, 0], [0, 0.6, 0.8]],
    dtype=np.float32,
)


def make_paper(arxiv_id: str = "2499.00001", version: int = 1) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        version=version,
        title="Judging the Judges: A Tiny Test Paper",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="We study how language models grade other models.",
        published=datetime(2024, 11, 23, tzinfo=UTC),
        updated=datetime(2025, 1, 2, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL", "cs.AI"],
    )


def store_paper(store, arxiv_id: str = "2499.00001", version: int = 1) -> list:
    chunks = chunk_paper(PARSED, arxiv_id, version)
    store.save_paper(make_paper(arxiv_id, version))
    store.replace_chunks(arxiv_id, version, chunks)
    return chunks


def store_vectors(store, chunks: list) -> None:
    store.ensure_vector_table(MODEL, 3)
    store.save_vectors(MODEL, [c.chunk_id for c in chunks], VECTORS)


def test_vector_table_name_is_a_safe_identifier_per_model():
    assert (
        vector_table("BAAI/bge-small-en-v1.5") == "chunk_vectors_baai_bge_small_en_v1_5"
    )
    assert (
        vector_table("Qwen/Qwen3-Embedding-0.6B")
        == "chunk_vectors_qwen_qwen3_embedding_0_6b"
    )
    with pytest.raises(ValueError, match="too long"):
        vector_table("org/" + "x" * 60)


def test_paper_round_trips_and_updates_in_place(store):
    store.save_paper(make_paper())
    assert store.get_paper("2499.00001", 1) == make_paper()

    store.save_paper(make_paper().model_copy(update={"title": "Renamed"}))
    assert store.get_paper("2499.00001", 1).title == "Renamed"


def test_unknown_paper_is_none(store):
    assert store.get_paper("2499.00001", 9) is None


def test_chunks_round_trip_in_order(store):
    store_paper(store)
    assert store.get_chunks("2499.00001", 1) == CHUNKS


def test_stored_chunker_version(store):
    assert store.stored_chunker_version("2499.00001", 1) is None
    store_paper(store)
    assert store.stored_chunker_version("2499.00001", 1) == CHUNKER_VERSION


def test_replacing_chunks_drops_the_old_chunks_and_their_vectors(store):
    store_vectors(store, store_paper(store))

    store.replace_chunks("2499.00001", 1, CHUNKS[:2])

    assert store.get_chunks("2499.00001", 1) == CHUNKS[:2]
    assert store.chunks_without_vectors("2499.00001", 1, MODEL) == CHUNKS[:2]


def test_replace_chunks_rejects_chunks_of_another_paper(store):
    store.save_paper(make_paper())
    with pytest.raises(ValueError, match="another paper"):
        store.replace_chunks("2499.00001", 2, CHUNKS)


def test_chunks_without_vectors_shrinks_as_vectors_are_saved(store):
    store_paper(store)
    store.ensure_vector_table(MODEL, 3)
    assert store.chunks_without_vectors("2499.00001", 1, MODEL) == CHUNKS

    store.save_vectors(MODEL, [c.chunk_id for c in CHUNKS[:4]], VECTORS[:4])
    assert store.chunks_without_vectors("2499.00001", 1, MODEL) == CHUNKS[4:]


def test_save_vectors_needs_one_vector_per_chunk(store):
    store.ensure_vector_table(MODEL, 3)
    with pytest.raises(ValueError, match="2 chunk ids but 6 vectors"):
        store.save_vectors(MODEL, ["a", "b"], VECTORS)


def test_vector_search_ranks_by_inner_product(store):
    store_vectors(store, store_paper(store))

    hits = store.vector_search(MODEL, np.array([1, 0, 0], dtype=np.float32), k=3)

    assert [hit.chunk.index for hit in hits] == [2, 4, 1]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.8, 0.6])
    assert hits[0].chunk == CHUNKS[2]


def test_vector_search_is_scoped_to_the_given_paper_versions(store):
    store_vectors(store, store_paper(store, "2499.00001", 1))
    store_vectors(store, store_paper(store, "2499.00001", 2))
    store_vectors(store, store_paper(store, "2499.00002", 1))
    query = np.array([1, 0, 0], dtype=np.float32)

    scoped = store.vector_search(MODEL, query, k=50, papers=[("2499.00001", 2)])
    everything = store.vector_search(MODEL, query, k=50)

    assert {(h.chunk.arxiv_id, h.chunk.version) for h in scoped} == {("2499.00001", 2)}
    assert len(scoped) == len(CHUNKS)
    assert len(everything) == 3 * len(CHUNKS)


def test_keyword_search_finds_the_chunk_with_the_words(store):
    store_paper(store)
    hits = store.keyword_search("Why does swapping the order help?")
    assert hits[0].chunk.index == 4


def test_keyword_search_needs_only_some_words_to_match(store):
    store_paper(store)
    hits = store.keyword_search("swapping pizza")
    assert [hit.chunk.index for hit in hits] == [4]


def test_keyword_search_ranks_more_matching_words_higher(store):
    store_paper(store)
    hits = store.keyword_search("GPT-4 agreement")
    assert hits[0].chunk.kind == "table"
    assert hits[0].score > hits[1].score


def test_keyword_search_with_only_stopwords_finds_nothing(store):
    store_paper(store)
    assert store.keyword_search("the of and") == []


def test_keyword_search_is_scoped_to_the_given_paper_versions(store):
    store_paper(store, "2499.00001", 1)
    store_paper(store, "2499.00002", 1)

    hits = store.keyword_search("swapping", papers=[("2499.00002", 1)])

    assert [(h.chunk.arxiv_id, h.chunk.index) for h in hits] == [("2499.00002", 4)]
