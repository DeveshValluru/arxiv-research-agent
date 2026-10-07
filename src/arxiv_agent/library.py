"""The local paper library, as agents use it: make sure a paper is indexed, then
pull out the passages that matter for a question.

Plain functions over our own database, not MCP tools: they're tied to this
index, so there's no other client to share them with.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from arxiv_agent.clients.arxiv import ArxivClient, ArxivError, PaperSummary
from arxiv_agent.ingestion.embedder import Embedder
from arxiv_agent.ingestion.html_cache import HTML_CACHE_DIR
from arxiv_agent.ingestion.pipeline import ingest_paper
from arxiv_agent.retrieval.retriever import Retriever
from arxiv_agent.storage.chunk_store import ChunkStore

Source = Literal["full_text", "abstract_only", "unavailable"]

logger = logging.getLogger(__name__)


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    section: str
    text: str


class Library:
    def __init__(
        self,
        store: ChunkStore,
        client: ArxivClient,
        embedder: Embedder,
        retriever: Retriever,
        count_tokens: Callable[[str], int],
        cache_dir: Path = HTML_CACHE_DIR,
    ) -> None:
        self._store = store
        self._client = client
        self._embedder = embedder
        self._retriever = retriever
        self._count_tokens = count_tokens
        self._cache_dir = cache_dir

    def ensure_ingested(self, arxiv_id: str, version: int) -> Source:
        # Cheap for a paper that's already in: every ingestion step skips work
        # that's done. A paper without an HTML version is still usable through
        # its abstract.
        try:
            paper = self._store.get_paper(arxiv_id, version) or self._lookup(
                arxiv_id, version
            )
            if paper is None:
                return "unavailable"
            report = ingest_paper(
                paper,
                self._client,
                self._store,
                self._embedder,
                self._count_tokens,
                self._cache_dir,
            )
        except ArxivError as exc:  # one paper failing shouldn't sink the review
            logger.warning("couldn't ingest %sv%d: %s", arxiv_id, version, exc)
            return "unavailable"
        return "full_text" if report.status == "ingested" else "abstract_only"

    def _lookup(self, arxiv_id: str, version: int) -> PaperSummary | None:
        paper = self._client.get_metadata([arxiv_id]).get(arxiv_id)
        if paper is None:
            return None
        # arXiv describes the latest version; titles and authors rarely change
        # between versions, so that stands in for the version requested.
        return paper.model_copy(update={"version": version})

    def passages(
        self, arxiv_id: str, version: int, question: str, k: int
    ) -> list[Passage]:
        # The abstract (what the paper claims overall), then the k chunks that
        # best match the question, without repeats.
        chunks = self._store.get_chunks(arxiv_id, version)
        if not chunks:
            paper = self._store.get_paper(arxiv_id, version)
            if paper is None:
                return []
            return [
                Passage(
                    chunk_id=f"{arxiv_id}v{version}:abstract",
                    section="Abstract",
                    text=paper.abstract,
                )
            ]
        abstract = [chunk for chunk in chunks if chunk.kind == "abstract"]
        hits = self._retriever.retrieve(question, (arxiv_id, version), k)
        picked = {c.chunk_id: c for c in [*abstract, *(hit.chunk for hit in hits)]}
        return [
            Passage(
                chunk_id=chunk.chunk_id,
                section=" > ".join(chunk.section_path) or "Abstract",
                text=chunk.text,
            )
            for chunk in picked.values()
        ]
