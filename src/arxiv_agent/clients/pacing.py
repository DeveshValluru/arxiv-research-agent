"""Taking turns with arXiv: one connection at a time, one request every 3 s.

arXiv's API terms set both limits per user, not per process. One process
pacing itself was enough while the review worker was the only client; with
the API (8.1) searching and ingesting while a review runs, and the worker's
arXiv MCP server in a subprocess of its own, three processes share the
budget. So the turn is taken in Postgres, which they all reach:
- an advisory lock: one request in flight at a time, across processes (and
  Postgres drops it if a holder dies)
- a timestamp row: the next request starts at least 3 s after the last one
  ended
Inside a process, threads also queue on a lock: advisory locks are
re-entrant per connection, so two threads sharing one would both get in.
Without a database (tests, a quick script), a process paces itself.
"""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

import psycopg
from psycopg.rows import tuple_row

MIN_INTERVAL = 3.0
ARXIV_LOCK = 5_202_002  # pg_advisory_lock key: holding it = talking to arXiv
SCHEMA = """
CREATE TABLE IF NOT EXISTS arxiv_pacing (
    id           integer PRIMARY KEY CHECK (id = 1),
    last_request timestamptz NOT NULL
);
"""


class Pacer(Protocol):
    def turn(self) -> Iterator[None]: ...


class LocalPacer:
    # This process only: what ArxivClient always did, now also safe across
    # threads.
    def __init__(
        self,
        interval: float = MIN_INTERVAL,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = interval
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    @contextmanager
    def turn(self) -> Iterator[None]:
        with self._lock:
            if self._last is not None:
                elapsed = self._clock() - self._last
                if elapsed < self._interval:
                    self._sleep(self._interval - elapsed)
            self._last = self._clock()
            yield


class SharedPacer:
    # Every process using this database takes turns.
    def __init__(
        self,
        conn: psycopg.Connection,
        interval: float = MIN_INTERVAL,
        sleep: Callable[[float], None] = time.sleep,
        key: int = ARXIV_LOCK,
    ) -> None:
        # conn: autocommit, used for nothing else (it holds the lock while a
        # request is in flight).
        self._conn = conn
        self._interval = interval
        self._sleep = sleep
        self._key = key
        self._threads = threading.Lock()
        self._conn.execute(SCHEMA)

    @classmethod
    def connect(cls, url: str) -> "SharedPacer":
        return cls(psycopg.connect(url, autocommit=True))

    @contextmanager
    def turn(self) -> Iterator[None]:
        with self._threads:
            self._conn.execute("SELECT pg_advisory_lock(%s)", (self._key,))
            try:
                with self._conn.cursor(row_factory=tuple_row) as cur:
                    row = cur.execute(
                        "SELECT extract(epoch FROM clock_timestamp() - last_request) "
                        "FROM arxiv_pacing"
                    ).fetchone()
                if row is not None and row[0] < self._interval:
                    self._sleep(self._interval - float(row[0]))
                yield
            finally:
                # Stamped when the request ends: the next one waits 3 s from
                # here, never less.
                self._conn.execute(
                    "INSERT INTO arxiv_pacing VALUES (1, clock_timestamp()) "
                    "ON CONFLICT (id) DO UPDATE SET last_request = clock_timestamp()"
                )
                self._conn.execute("SELECT pg_advisory_unlock(%s)", (self._key,))
