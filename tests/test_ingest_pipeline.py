from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from arxiv_agent.clients.arxiv import PaperSummary
from arxiv_agent.ingestion.chunker import CHUNKER_VERSION, _approx_tokens, chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.ingestion.pipeline import ingest_paper, load_html

FIXTURES = Path(__file__).parent / "fixtures"
HTML = (FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")
CHUNKS = chunk_paper(parse_arxiv_html(HTML), "2499.00001", 1)
PAPER = PaperSummary(
    arxiv_id="2499.00001",
    version=1,
    title="Judging the Judges: A Tiny Test Paper",
    authors=["Ada Lovelace"],
    abstract="We study how language models grade other models.",
    published=datetime(2024, 11, 23, tzinfo=UTC),
    updated=datetime(2024, 11, 23, tzinfo=UTC),
    primary_category="cs.CL",
    categories=["cs.CL"],
)


class FakeClient:
    def __init__(self, html: str | None) -> None:
        self.html = html
        self.fetched: list[tuple[str, int | None]] = []

    def fetch_html(self, arxiv_id: str, version: int | None = None) -> str | None:
        self.fetched.append((arxiv_id, version))
        return self.html


class FakeEmbedder:
    dimension = 3

    def __init__(self, model_id: str = "test/fake-embedder") -> None:
        self.model_id = model_id
        self.embedded: list[str] = []

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.embedded.extend(texts)
        return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1))


def ingest(store, client, embedder, cache_dir):
    return ingest_paper(
        PAPER, client, store, embedder, count_tokens=_approx_tokens, cache_dir=cache_dir
    )


def test_load_html_reads_the_cache_without_calling_arxiv(tmp_path):
    (tmp_path / "2499.00001v1.html").write_text("<cached>", encoding="utf-8")
    client = FakeClient("<fresh>")

    assert load_html(client, "2499.00001", 1, tmp_path) == "<cached>"
    assert client.fetched == []


def test_load_html_fetches_once_and_caches(tmp_path):
    client = FakeClient("<fresh>")

    assert load_html(client, "2499.00001", 1, tmp_path) == "<fresh>"
    assert load_html(client, "2499.00001", 1, tmp_path) == "<fresh>"
    assert client.fetched == [("2499.00001", 1)]


def test_load_html_does_not_cache_a_missing_page(tmp_path):
    assert load_html(FakeClient(None), "2499.00001", 1, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_load_html_handles_old_style_ids(tmp_path):
    load_html(FakeClient("<old>"), "hep-th/9901001", 2, tmp_path)
    assert (tmp_path / "hep-th_9901001v2.html").read_text(encoding="utf-8") == "<old>"


def test_ingest_stores_paper_chunks_and_vectors(store, tmp_path):
    report = ingest(store, FakeClient(HTML), FakeEmbedder(), tmp_path)

    assert report.status == "ingested"
    assert report.chunks_written == len(CHUNKS)
    assert report.vectors_written == len(CHUNKS)
    assert store.get_paper("2499.00001", 1) == PAPER
    assert store.get_chunks("2499.00001", 1) == CHUNKS
    assert store.chunks_without_vectors("2499.00001", 1, "test/fake-embedder") == []


def test_second_ingest_does_no_work(store, tmp_path):
    client, embedder = FakeClient(HTML), FakeEmbedder()
    ingest(store, client, embedder, tmp_path)

    report = ingest(store, client, embedder, tmp_path)

    assert (report.chunks_written, report.vectors_written) == (0, 0)
    assert len(client.fetched) == 1
    assert len(embedder.embedded) == len(CHUNKS)


def test_new_embedder_reuses_the_stored_chunks(store, tmp_path):
    ingest(store, FakeClient(HTML), FakeEmbedder(), tmp_path)
    other = FakeEmbedder("test/other-embedder")

    report = ingest(store, FakeClient(HTML), other, tmp_path)

    assert (report.chunks_written, report.vectors_written) == (0, len(CHUNKS))
    assert other.embedded == [c.embed_text for c in CHUNKS]


def test_stale_chunker_version_rebuilds_chunks_and_vectors(store, tmp_path):
    old = [
        c.model_copy(update={"chunker_version": CHUNKER_VERSION - 1}) for c in CHUNKS
    ]
    store.save_paper(PAPER)
    store.replace_chunks("2499.00001", 1, old)
    store.ensure_vector_table("test/fake-embedder", 3)
    store.save_vectors(
        "test/fake-embedder", [c.chunk_id for c in old], np.ones((len(old), 3))
    )

    report = ingest(store, FakeClient(HTML), FakeEmbedder(), tmp_path)

    assert (report.chunks_written, report.vectors_written) == (len(CHUNKS), len(CHUNKS))
    assert store.stored_chunker_version("2499.00001", 1) == CHUNKER_VERSION


def test_paper_without_html_keeps_its_metadata_but_no_chunks(store, tmp_path):
    report = ingest(store, FakeClient(None), FakeEmbedder(), tmp_path)

    assert report.status == "no_html"
    assert store.get_paper("2499.00001", 1) == PAPER
    assert store.get_chunks("2499.00001", 1) == []
