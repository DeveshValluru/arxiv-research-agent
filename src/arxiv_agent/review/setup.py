"""Build a ready-to-run review graph with every real dependency.

Shared by scripts/review.py (one review in the foreground) and the worker (a
review per job). Building is the slow part (models load, MCP servers start),
so the worker builds once and reuses the graph for every job.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langfuse import Langfuse
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict

from arxiv_agent.clients.arxiv import ArxivClient
from arxiv_agent.evals.judge import JUDGE_MODEL
from arxiv_agent.ingestion.chunker import CHUNK_TOKENIZER
from arxiv_agent.ingestion.embedder import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL_ID,
    Embedder,
    load_token_counter,
)
from arxiv_agent.library import Library
from arxiv_agent.llm import ChatModel
from arxiv_agent.retrieval.retriever import build_retriever
from arxiv_agent.review.critic import Critic
from arxiv_agent.review.graph import build_review_graph
from arxiv_agent.review.human import HumanReview
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.reader import Reader
from arxiv_agent.review.writer import Synthesizer
from arxiv_agent.storage.chunk_store import ChunkStore
from arxiv_agent.tools.toolbox import McpToolbox


class ReviewSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "Qwen/Qwen3-32B"
    providers: list[str] = ["deepinfra", "nscale"]
    # The judge is from a different model family than the writer. Fastest
    # first, measured with 4 parallel calls: together < 1 s, novita ~2 s,
    # ovhcloud ~3 s. (The eval judge keeps its own order, so eval scores stay
    # comparable.)
    judge_providers: list[str] = ["together", "novita", "ovhcloud"]
    keep: int = 8
    max_revisions: int = 2


@asynccontextmanager
async def open_review_graph(
    settings: ReviewSettings, langfuse: Langfuse, checkpointer: BaseCheckpointSaver
) -> AsyncIterator[object]:
    # checkpointer: Postgres for the worker (jobs survive restarts and pauses),
    # memory for a foreground review.
    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    try:
        embedder = Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
        library = Library(
            store,
            ArxivClient(),
            embedder,
            build_retriever("rerank", store, embedder),
            load_token_counter(CHUNK_TOKENIZER),
        )
        writer = ChatModel(
            settings.model, settings.providers, langfuse=langfuse, timeout=120
        )
        # Judge calls are short (~2 s): a provider taking 30 s counts as down.
        judge = ChatModel(
            JUDGE_MODEL, settings.judge_providers, langfuse=langfuse, timeout=30
        )
        async with McpToolbox(langfuse=langfuse) as toolbox:  # arXiv + OpenAlex
            yield build_review_graph(
                ReviewNodes(writer, toolbox, keep=settings.keep, langfuse=langfuse),
                Reader(writer, library, langfuse=langfuse),
                Synthesizer(writer, langfuse=langfuse),
                Critic(judge, max_revisions=settings.max_revisions, langfuse=langfuse),
                HumanReview(toolbox, langfuse=langfuse),
                checkpointer=checkpointer,
            )
    finally:
        store.close()
