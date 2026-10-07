"""Review jobs in Postgres: a queue the worker takes jobs from, and an event log
anyone can read to follow a job, and pick it up again after disconnecting.

A job's events are numbered 1, 2, 3...; a watcher remembers the last number it
saw and asks for the ones after it. That's what makes reconnecting work: the
SSE endpoint will do the same with Last-Event-ID.
"""

import uuid
from datetime import datetime
from typing import Literal

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_jobs (
    job_id       uuid PRIMARY KEY,
    question     text NOT NULL,
    budget       jsonb NOT NULL,
    status       text NOT NULL
                 CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts     integer NOT NULL DEFAULT 0,
    created_at   timestamptz NOT NULL DEFAULT now(),
    started_at   timestamptz,
    heartbeat_at timestamptz,
    finished_at  timestamptz,
    trace_id     text,
    error        text,
    result       jsonb
);
CREATE INDEX IF NOT EXISTS review_jobs_queue
    ON review_jobs (created_at) WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS review_job_events (
    job_id uuid NOT NULL REFERENCES review_jobs ON DELETE CASCADE,
    seq    integer NOT NULL,
    at     timestamptz NOT NULL DEFAULT now(),
    kind   text NOT NULL,
    data   jsonb NOT NULL,
    PRIMARY KEY (job_id, seq)
);
"""
# Everything but the result, which can be hundreds of KB: fetched on its own.
JOB_COLUMNS = """job_id::text AS job_id, question, budget, status, attempts,
    created_at, started_at, heartbeat_at, finished_at, trace_id, error"""
# Key for pg_try_advisory_lock: whoever holds it is the one review worker.
WORKER_LOCK = 5_202_001

Status = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    question: str
    budget: dict
    status: Status
    attempts: int
    created_at: datetime
    started_at: datetime | None
    heartbeat_at: datetime | None
    finished_at: datetime | None
    trace_id: str | None
    error: str | None


class JobEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    at: datetime
    kind: str  # "status", "step" or "progress"
    data: dict


class JobStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    @classmethod
    def connect(cls, url: str) -> "JobStore":
        store = cls(psycopg.connect(url, autocommit=True, row_factory=dict_row))
        store.init_schema()
        return store

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        self._conn.execute(SCHEMA)

    # --- submitting and reading ---------------------------------------------------

    def submit(self, question: str, budget: dict) -> str:
        job_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO review_jobs (job_id, question, budget, status) "
            "VALUES (%s, %s, %s, 'queued')",
            (job_id, question, Jsonb(budget)),
        )
        self.add_event(job_id, "status", {"status": "queued"})
        return job_id

    def get(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            f"SELECT {JOB_COLUMNS} FROM review_jobs WHERE job_id = %s", (job_id,)
        ).fetchone()
        return Job(**row) if row else None

    def recent(self, limit: int = 10) -> list[Job]:
        rows = self._conn.execute(
            f"SELECT {JOB_COLUMNS} FROM review_jobs ORDER BY created_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
        return [Job(**row) for row in rows]

    def result(self, job_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT result FROM review_jobs WHERE job_id = %s", (job_id,)
        ).fetchone()
        return row["result"] if row else None

    # --- events -----------------------------------------------------------------

    def add_event(self, job_id: str, kind: str, data: dict) -> int:
        # Numbered per job. Only one process writes a job's events at a time
        # (whoever submitted it, then its worker), so MAX + 1 can't race; the
        # primary key would reject a duplicate if it ever did.
        row = self._conn.execute(
            """
            INSERT INTO review_job_events (job_id, seq, kind, data)
            SELECT %(job_id)s, COALESCE(MAX(seq), 0) + 1, %(kind)s, %(data)s
            FROM review_job_events WHERE job_id = %(job_id)s
            RETURNING seq
            """,
            {"job_id": job_id, "kind": kind, "data": Jsonb(data)},
        ).fetchone()
        return row["seq"]

    def events_after(self, job_id: str, seq: int = 0) -> list[JobEvent]:
        rows = self._conn.execute(
            "SELECT seq, at, kind, data FROM review_job_events "
            "WHERE job_id = %s AND seq > %s ORDER BY seq",
            (job_id, seq),
        ).fetchall()
        return [JobEvent(**row) for row in rows]

    # --- the worker's side --------------------------------------------------------

    def try_lock_worker(self, key: int = WORKER_LOCK) -> bool:
        # A session-level advisory lock: held while this connection lives, and
        # released by Postgres if the worker process dies.
        row = self._conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (key,))
        return row.fetchone()["ok"]

    def claim_next(self) -> Job | None:
        # FOR UPDATE SKIP LOCKED: two workers can't claim the same job, and
        # neither waits on a row the other is in the middle of claiming.
        row = self._conn.execute(
            f"""
            UPDATE review_jobs
            SET status = 'running', attempts = attempts + 1,
                started_at = now(), heartbeat_at = now()
            WHERE job_id = (
                SELECT job_id FROM review_jobs WHERE status = 'queued'
                ORDER BY created_at LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING {JOB_COLUMNS}
            """
        ).fetchone()
        if row is None:
            return None
        job = Job(**row)
        self.add_event(
            job.job_id, "status", {"status": "running", "attempt": job.attempts}
        )
        return job

    def heartbeat(self, job_id: str) -> None:
        self._conn.execute(
            "UPDATE review_jobs SET heartbeat_at = now() WHERE job_id = %s",
            (job_id,),
        )

    def finish(self, job_id: str, result: dict, trace_id: str | None) -> None:
        self._conn.execute(
            "UPDATE review_jobs SET status = 'done', finished_at = now(), "
            "result = %s, trace_id = %s WHERE job_id = %s",
            (Jsonb(result), trace_id, job_id),
        )
        self.add_event(job_id, "status", {"status": "done"})

    def fail(self, job_id: str, error: str) -> None:
        self._conn.execute(
            "UPDATE review_jobs SET status = 'failed', finished_at = now(), "
            "error = %s WHERE job_id = %s",
            (error, job_id),
        )
        self.add_event(job_id, "status", {"status": "failed", "error": error})

    def recover_orphans(self, max_attempts: int = 2) -> dict[str, Status]:
        # Called by the worker once it holds the worker lock: a job still
        # 'running' then belongs to a worker that died mid-job. It goes back in
        # the queue, unless it has already had max_attempts tries.
        rows = self._conn.execute(
            """
            UPDATE review_jobs
            SET status = CASE WHEN attempts >= %(max)s THEN 'failed' ELSE 'queued' END,
                finished_at = CASE WHEN attempts >= %(max)s THEN now() END,
                error = CASE WHEN attempts >= %(max)s
                        THEN 'the worker stopped during each of '
                             || attempts || ' attempts' END
            WHERE status = 'running'
            RETURNING job_id::text AS job_id, status, error
            """,
            {"max": max_attempts},
        ).fetchall()
        for row in rows:
            data = {"status": row["status"], "after": "the worker stopped mid-job"}
            if row["error"]:
                data["error"] = row["error"]
            self.add_event(row["job_id"], "status", data)
        return {row["job_id"]: row["status"] for row in rows}
