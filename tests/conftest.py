import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from arxiv_agent.storage.chunk_store import ChunkStore
from arxiv_agent.storage.job_store import JobStore

# Tests must never send traces to the real Langfuse project, even when run with
# --env-file .env (which holds the real keys). The tracing code still runs; the
# client just doesn't export anything.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"


@pytest.fixture
def connect():
    # Each test gets its own throwaway schema in the dev database, so tests never
    # touch the real index. connect() opens a connection that uses it; a test can
    # open several (two workers, say). Run with: uv run --env-file .env python -m pytest
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set: run pytest with --env-file .env")
    try:
        admin = psycopg.connect(url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running: docker compose up -d")

    schema = sql.Identifier(f"test_{uuid.uuid4().hex[:12]}")
    admin.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
    opened: list[psycopg.Connection] = []

    def open_connection() -> psycopg.Connection:
        conn = psycopg.connect(
            url, autocommit=True, row_factory=dict_row, connect_timeout=3
        )
        conn.execute(sql.SQL("SET search_path TO {}, public").format(schema))
        opened.append(conn)
        return conn

    yield open_connection

    for conn in opened:
        conn.close()
    admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
    admin.close()


@pytest.fixture
def store(connect):
    store = ChunkStore(connect())
    store.init_schema()
    return store


@pytest.fixture
def jobs(connect):
    jobs = JobStore(connect())
    jobs.init_schema()
    return jobs
