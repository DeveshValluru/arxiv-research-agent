"""What the API's routes use, built once when the server starts.

The models (embedder, reranker) take seconds to load, so they load at
startup, not on the first question. Each store has its own connection;
psycopg runs one statement at a time per connection, which is plenty for a
local demo (a server with real traffic would use a pool).
"""

import os
from dataclasses import dataclass

import psycopg
from langfuse import get_client
from psycopg.rows import dict_row

from arxiv_agent.clients.arxiv import ArxivClient, shared_arxiv_client
from arxiv_agent.evals.qa_run import MODEL, PROVIDERS
from arxiv_agent.ingestion.chunker import CHUNK_TOKENIZER
from arxiv_agent.ingestion.embedder import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL_ID,
    Embedder,
    load_token_counter,
)
from arxiv_agent.library import Library
from arxiv_agent.qa.answerer import Answerer
from arxiv_agent.qa.support import SupportChecker
from arxiv_agent.retrieval.retriever import build_retriever
from arxiv_agent.storage.chunk_store import ChunkStore
from arxiv_agent.storage.job_store import JobStore
from arxiv_agent.storage.reading_list import ReadingList


@dataclass
class Services:
    chunks: ChunkStore
    jobs: JobStore
    reading: ReadingList
    arxiv: ArxivClient
    library: Library
    answerer: Answerer

    def close(self) -> None:
        self.chunks.close()
        self.jobs.close()


def build_services() -> Services:
    url = os.environ["DATABASE_URL"]
    # Real use, like ask.py: what the flywheel harvests flags from (7.3).
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    langfuse = get_client()
    chunks = ChunkStore.connect(url)
    arxiv = shared_arxiv_client()
    embedder = Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
    # One retriever for both, so ingesting a paper refreshes what the
    # Answerer searches (Library.ensure_ingested calls forget()).
    retriever = build_retriever("rerank", chunks, embedder)
    return Services(
        chunks=chunks,
        jobs=JobStore.connect(url),
        reading=ReadingList(
            psycopg.connect(url, autocommit=True, row_factory=dict_row)
        ),
        arxiv=arxiv,
        library=Library(
            chunks, arxiv, embedder, retriever, load_token_counter(CHUNK_TOKENIZER)
        ),
        answerer=Answerer(
            retriever,
            model=MODEL,
            providers=PROVIDERS.split(","),
            langfuse=langfuse,
            support=SupportChecker(langfuse=langfuse),
        ),
    )
