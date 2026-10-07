from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from arxiv_agent.clients.arxiv import ArxivUnavailableError, PaperSummary
from arxiv_agent.ingestion.chunker import _approx_tokens
from arxiv_agent.library import Library, Passage
from arxiv_agent.storage.chunk_store import SearchHit

HTML = (Path(__file__).parent / "fixtures" / "latexml_minimal.html").read_text(
    encoding="utf-8"
)
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
    def __init__(self, html=HTML, papers=(PAPER,), error=None) -> None:
        self.html = html
        self.papers = {paper.arxiv_id: paper for paper in papers}
        self.error = error
        self.lookups: list[list[str]] = []
        self.fetched: list[tuple[str, int | None]] = []

    def get_metadata(self, ids):
        self.lookups.append(ids)
        if self.error:
            raise self.error
        return {i: self.papers[i] for i in ids if i in self.papers}

    def fetch_html(self, arxiv_id, version=None):
        self.fetched.append((arxiv_id, version))
        return self.html


class FakeEmbedder:
    model_id = "test/fake-embedder"
    dimension = 3

    def embed_passages(self, texts):
        return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1))


class LastChunks:
    # Stands in for the hybrid retriever: the paper's last k chunks.
    name = "last-chunks"

    def __init__(self, store) -> None:
        self.store = store

    def retrieve(self, question, paper, k):
        chunks = self.store.get_chunks(*paper)
        return [SearchHit(chunk=chunk, score=1.0) for chunk in chunks[::-1][:k]]


def make_library(store, client, cache_dir) -> Library:
    return Library(
        store, client, FakeEmbedder(), LastChunks(store), _approx_tokens, cache_dir
    )


def test_a_new_paper_is_looked_up_ingested_and_searchable(store, tmp_path):
    client = FakeClient()
    library = make_library(store, client, tmp_path)

    assert library.ensure_ingested("2499.00001", 1) == "full_text"
    passages = library.passages("2499.00001", 1, "position bias?", k=2)

    assert client.lookups == [["2499.00001"]]
    # The abstract always comes first, then the retrieved chunks.
    assert [p.chunk_id for p in passages] == [
        "2499.00001v1:0000",
        "2499.00001v1:0005",
        "2499.00001v1:0004",
    ]
    assert passages[0].section == "Abstract"
    assert passages[2].section == "2. Method > 2.1. Position Bias > 2.1.1. Swapping"


def test_a_stored_paper_is_not_looked_up_or_fetched_again(store, tmp_path):
    client = FakeClient()
    library = make_library(store, client, tmp_path)

    library.ensure_ingested("2499.00001", 1)
    library.ensure_ingested("2499.00001", 1)

    assert client.lookups == [["2499.00001"]]
    assert client.fetched == [("2499.00001", 1)]


def test_the_abstract_is_not_repeated_when_it_is_also_retrieved(store, tmp_path):
    library = make_library(store, FakeClient(), tmp_path)
    library.ensure_ingested("2499.00001", 1)

    passages = library.passages("2499.00001", 1, "anything", k=10)

    ids = [p.chunk_id for p in passages]
    assert ids[0] == "2499.00001v1:0000"
    assert len(ids) == len(set(ids)) == 6


def test_a_paper_without_html_is_read_from_its_abstract(store, tmp_path):
    library = make_library(store, FakeClient(html=None), tmp_path)

    assert library.ensure_ingested("2499.00001", 1) == "abstract_only"
    assert library.passages("2499.00001", 1, "q", k=3) == [
        Passage(
            chunk_id="2499.00001v1:abstract", section="Abstract", text=PAPER.abstract
        )
    ]


def test_an_unknown_paper_is_unavailable(store, tmp_path):
    library = make_library(store, FakeClient(papers=()), tmp_path)

    assert library.ensure_ingested("2499.00001", 1) == "unavailable"
    assert library.passages("2499.00001", 1, "q", k=3) == []


def test_an_arxiv_outage_makes_a_paper_unavailable_not_an_error(store, tmp_path):
    client = FakeClient(error=ArxivUnavailableError("down"))
    library = make_library(store, client, tmp_path)

    assert library.ensure_ingested("2499.00001", 1) == "unavailable"


def test_latest_metadata_stands_in_for_an_older_version(store, tmp_path):
    client = FakeClient(papers=(PAPER.model_copy(update={"version": 3}),))
    library = make_library(store, client, tmp_path)

    assert library.ensure_ingested("2499.00001", 1) == "full_text"
    assert store.get_paper("2499.00001", 1).title == PAPER.title
    assert client.fetched == [("2499.00001", 1)]


def test_a_page_that_cant_be_parsed_makes_the_paper_unavailable_not_an_error(
    store, tmp_path
):
    library = make_library(store, FakeClient(html="<html>not a paper</html>"), tmp_path)
    assert library.ensure_ingested("2499.00001", 1) == "unavailable"
