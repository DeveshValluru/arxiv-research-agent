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
    job_id           uuid PRIMARY KEY,
    question         text NOT NULL,
    budget           jsonb NOT NULL,
    pause_for_review boolean NOT NULL DEFAULT false,
    status           text NOT NULL,
    attempts         integer NOT NULL DEFAULT 0,
    created_at       timestamptz NOT NULL DEFAULT now(),
    started_at       timestamptz,
    heartbeat_at     timestamptz,
    paused_at        timestamptz,
    finished_at      timestamptz,
    trace_id         text,
    error            text,
    review_request   jsonb,  -- what the person sees at the pause
    decision         jsonb,  -- what they decided
    result           jsonb
);
-- Databases created before the pause existed (5.2b) get its columns, and the
-- status check is (re)made with every status.
ALTER TABLE review_jobs ADD COLUMN IF NOT EXISTS pause_for_review boolean NOT NULL DEFAULT false;
ALTER TABLE review_jobs ADD COLUMN IF NOT EXISTS paused_at timestamptz;
ALTER TABLE review_jobs ADD COLUMN IF NOT EXISTS review_request jsonb;
ALTER TABLE review_jobs ADD COLUMN IF NOT EXISTS decision jsonb;
ALTER TABLE review_jobs DROP CONSTRAINT IF EXISTS review_jobs_status_check;
ALTER TABLE review_jobs ADD CONSTRAINT review_jobs_status_check CHECK (status IN
    ('queued', 'running', 'awaiting_review', 'done', 'failed', 'expired'));
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

-- The reviewer's edits as eval labels (see review.state.Edit), one per paper.
CREATE TABLE IF NOT EXISTS review_labels (
    job_id         uuid NOT NULL REFERENCES review_jobs ON DELETE CASCADE,
    arxiv_id       text NOT NULL,
    label          text NOT NULL,
    question       text NOT NULL,
    title          text NOT NULL,
    screener_score integer,
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, arxiv_id)
);
"""
# Everything but the result, which can be hundreds of KB: fetched on its own.
JOB_COLUMNS = """job_id::text AS job_id, question, budget, pause_for_review,
    status, attempts, created_at, started_at, heartbeat_at, paused_at,
    finished_at, trace_id, error, decision"""
# Key for pg_try_advisory_lock: whoever holds it is the one review worker.
WORKER_LOCK = 5_202_001

Status = Literal["queued", "running", "awaiting_review", "done", "failed", "expired"]


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    question: str
    budget: dict
    pause_for_review: bool
    status: Status
    attempts: int  # tries at the current run; a review decision starts a new run
    created_at: datetime
    started_at: datetime | None
    heartbeat_at: datetime | None
    paused_at: datetime | None
    finished_at: datetime | None
    trace_id: str | None
    error: str | None
    decision: dict | None


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

    def submit(
        self, question: str, budget: dict, pause_for_review: bool = False
    ) -> str:
        job_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO review_jobs (job_id, question, budget, pause_for_review, status) "
            "VALUES (%s, %s, %s, %s, 'queued')",
            (job_id, question, Jsonb(budget), pause_for_review),
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

    def worker_running(self, key: int = WORKER_LOCK) -> bool:
        # Whether some session holds the worker's lock, i.e. a worker is up.
        # pg_locks splits a bigint key: high 32 bits in classid, low in objid.
        row = self._conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
            "AND classid::bigint = %s AND objid::bigint = %s AND objsubid = 1 "
            "AND granted) AS held",
            (key >> 32, key & 0xFFFFFFFF),
        ).fetchone()
        return row["held"]

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
        data = {"status": "running", "attempt": job.attempts}
        if job.decision is not None:
            data["after"] = "your review"
        self.add_event(job.job_id, "status", data)
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

    def retry(self, job_id: str) -> bool:
        # A failed job goes back in the queue; the worker continues it from
        # its last checkpoint rather than from the start.
        row = self._conn.execute(
            "UPDATE review_jobs SET status = 'queued', attempts = 0, error = NULL, "
            "finished_at = NULL WHERE job_id = %s AND status = 'failed' RETURNING 1",
            (job_id,),
        ).fetchone()
        if row:
            self.add_event(job_id, "status", {"status": "queued", "after": "a retry"})
        return row is not None

    # --- the pause for review ------------------------------------------------------

    def pause(self, job_id: str, request: dict) -> None:
        self._conn.execute(
            "UPDATE review_jobs SET status = 'awaiting_review', paused_at = now(), "
            "review_request = %s WHERE job_id = %s",
            (Jsonb(request), job_id),
        )
        self.add_event(
            job_id,
            "status",
            {"status": "awaiting_review", "kept": len(request["kept"])},
        )

    def review_request(self, job_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT review_request FROM review_jobs WHERE job_id = %s", (job_id,)
        ).fetchone()
        return row["review_request"] if row else None

    def decide(self, job_id: str, decision: dict) -> bool:
        # Only a job that is waiting can take a decision. It goes back in the
        # queue as a new run: attempts start again from zero.
        row = self._conn.execute(
            "UPDATE review_jobs SET status = 'queued', decision = %s, attempts = 0 "
            "WHERE job_id = %s AND status = 'awaiting_review' RETURNING 1",
            (Jsonb(decision), job_id),
        ).fetchone()
        if row:
            self.add_event(
                job_id,
                "status",
                {
                    "status": "queued",
                    "removing": len(decision.get("remove", [])),
                    "adding": len(decision.get("add", [])),
                },
            )
        return row is not None

    def expire_reviews(self, max_hours: float = 24) -> list[str]:
        # A pause can't hold a review forever: after max_hours it's dropped.
        rows = self._conn.execute(
            """
            UPDATE review_jobs SET status = 'expired', finished_at = now(),
                error = 'no review within ' || %(hours)s || ' hours'
            WHERE status = 'awaiting_review'
              AND paused_at < now() - make_interval(secs => %(hours)s * 3600)
            RETURNING job_id::text AS job_id
            """,
            {"hours": max_hours},
        ).fetchall()
        for row in rows:
            self.add_event(row["job_id"], "status", {"status": "expired"})
        return [row["job_id"] for row in rows]

    # --- labels ----------------------------------------------------------------------

    def save_labels(self, job_id: str, question: str, edits: list[dict]) -> None:
        # Upsert: saving the same edits again (a retried run) changes nothing.
        for edit in edits:
            self._conn.execute(
                """
                INSERT INTO review_labels
                    (job_id, arxiv_id, label, question, title, screener_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, arxiv_id) DO UPDATE SET
                    label = EXCLUDED.label, screener_score = EXCLUDED.screener_score
                """,
                (
                    job_id,
                    edit["arxiv_id"],
                    edit["label"],
                    question,
                    edit["title"],
                    edit["screener_score"],
                ),
            )

    def labels(self, limit: int = 50) -> list[dict]:
        return self._conn.execute(
            "SELECT job_id::text AS job_id, arxiv_id, label, question, title, "
            "screener_score, created_at FROM review_labels "
            "ORDER BY created_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
