from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.clients.arxiv import ArxivClient, PaperSummary
from arxiv_agent.ingestion.chunker import CHUNKER_VERSION, chunk_paper
from arxiv_agent.ingestion.embedder import Embedder
from arxiv_agent.ingestion.html_cache import HTML_CACHE_DIR, load_html
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.storage.chunk_store import ChunkStore


class IngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: int
    status: Literal["ingested", "no_html"]
    chunks_written: int = 0
    vectors_written: int = 0


def ingest_paper(
    paper: PaperSummary,
    client: ArxivClient,
    store: ChunkStore,
    embedder: Embedder,
    count_tokens: Callable[[str], int],
    cache_dir: Path = HTML_CACHE_DIR,
) -> IngestReport:
    # Each step does only the work that is missing, so re-running is cheap and
    # an interrupted run picks up where it stopped.
    arxiv_id, version = paper.arxiv_id, paper.version
    store.save_paper(paper)

    chunks_written = 0
    if store.stored_chunker_version(arxiv_id, version) != CHUNKER_VERSION:
        html = load_html(client, arxiv_id, version, cache_dir)
        if html is None:
            return IngestReport(arxiv_id=arxiv_id, version=version, status="no_html")
        chunks = chunk_paper(
            parse_arxiv_html(html), arxiv_id, version, count_tokens=count_tokens
        )
        store.replace_chunks(arxiv_id, version, chunks)
        chunks_written = len(chunks)

    store.ensure_vector_table(embedder.model_id, embedder.dimension)
    missing = store.chunks_without_vectors(arxiv_id, version, embedder.model_id)
    if missing:
        vectors = embedder.embed_passages([c.embed_text for c in missing])
        store.save_vectors(embedder.model_id, [c.chunk_id for c in missing], vectors)

    return IngestReport(
        arxiv_id=arxiv_id,
        version=version,
        status="ingested",
        chunks_written=chunks_written,
        vectors_written=len(missing),
    )
