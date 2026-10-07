"""Where the review graph saves its state: Postgres for jobs, memory for tests
and the foreground script.

Two things here aren't LangGraph defaults:
- An allowlist of the types a checkpoint may contain. Loading a checkpoint
  rebuilds Python objects from database rows; without a list, anyone who can
  write those rows could make us build any class. LangGraph warns about this
  and will soon require it.
- An async adapter for the sync Postgres saver. The async saver needs a
  "selector" event loop, which on Windows can't start subprocesses, and our MCP
  servers are subprocesses. So the sync saver runs in worker threads instead,
  the same way ChatModel runs its blocking HTTP calls.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row

from arxiv_agent.llm import Usage
from arxiv_agent.review.state import (
    Budget,
    Candidate,
    Claim,
    Critique,
    Edit,
    Evidence,
    ReadReport,
    ScreenedPaper,
    SentenceCheck,
)

STATE_TYPES = (
    Budget,
    Usage,
    Candidate,
    ScreenedPaper,
    Edit,
    Claim,
    ReadReport,
    SentenceCheck,
    Critique,
    Evidence,
)


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in STATE_TYPES]
    )


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=serializer())


class ThreadedPostgresSaver(PostgresSaver):
    # The async methods LangGraph calls, each running its sync twin in a
    # thread. PostgresSaver already serializes access to its connection with
    # a lock, so calls from several threads are safe.
    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self, config, *, filter=None, before=None, limit=None
    ) -> AsyncIterator:
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(
            self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)

    async def aget_delta_channel_history(self, *, config, channels):
        return await asyncio.to_thread(
            self.get_delta_channel_history, config=config, channels=channels
        )


@contextmanager
def postgres_checkpointer(url: str) -> Iterator[ThreadedPostgresSaver]:
    # The connection settings PostgresSaver requires.
    with psycopg.connect(
        url, autocommit=True, row_factory=dict_row, prepare_threshold=0
    ) as conn:
        saver = ThreadedPostgresSaver(conn, serde=serializer())
        saver.setup()  # creates or migrates its tables
        yield saver
