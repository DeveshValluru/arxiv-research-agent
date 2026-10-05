import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from arxiv_agent.storage.chunk_store import ChunkStore


@pytest.fixture
def store():
    # Each test gets its own throwaway schema in the dev database, so tests never
    # touch the real index. Run with: uv run --env-file .env python -m pytest
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set: run pytest with --env-file .env")
    try:
        conn = psycopg.connect(
            url, autocommit=True, row_factory=dict_row, connect_timeout=3
        )
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running: docker compose up -d")

    schema = sql.Identifier(f"test_{uuid.uuid4().hex[:12]}")
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
    conn.execute(sql.SQL("SET search_path TO {}, public").format(schema))
    store = ChunkStore(conn)
    store.init_schema()

    yield store

    conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
    conn.close()
