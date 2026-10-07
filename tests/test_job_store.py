import random

from arxiv_agent.storage.job_store import JobStore

BUDGET = {"max_llm_calls": 100, "max_tokens": 150_000, "max_seconds": 600}


def kinds(jobs, job_id) -> list[tuple[str, dict]]:
    return [(e.kind, e.data) for e in jobs.events_after(job_id)]


def test_a_submitted_job_waits_in_the_queue(jobs):
    job_id = jobs.submit("How biased are LLM judges?", BUDGET)

    job = jobs.get(job_id)
    assert (job.status, job.attempts, job.budget) == ("queued", 0, BUDGET)
    assert kinds(jobs, job_id) == [("status", {"status": "queued"})]
    assert jobs.result(job_id) is None
    assert jobs.get("00000000-0000-0000-0000-000000000000") is None


def test_jobs_are_claimed_oldest_first_and_only_once(jobs):
    first = jobs.submit("first", BUDGET)
    second = jobs.submit("second", BUDGET)

    claimed = jobs.claim_next()
    assert claimed.job_id == first
    assert (claimed.status, claimed.attempts) == ("running", 1)
    assert claimed.started_at is not None and claimed.heartbeat_at is not None
    assert jobs.claim_next().job_id == second
    assert jobs.claim_next() is None
    assert kinds(jobs, first)[-1] == ("status", {"status": "running", "attempt": 1})


def test_a_job_another_worker_is_claiming_is_skipped_not_waited_for(connect):
    worker_a, worker_b = connect(), connect()
    jobs_a, jobs_b = JobStore(worker_a), JobStore(worker_b)
    jobs_a.init_schema()
    first = jobs_a.submit("first", BUDGET)
    second = jobs_a.submit("second", BUDGET)
    # Fail instead of hanging if SKIP LOCKED were missing.
    worker_b.execute("SET lock_timeout = '2s'")

    with worker_a.transaction():  # A holds the lock on the first job's row
        worker_a.execute(
            "SELECT 1 FROM review_jobs WHERE job_id = %s FOR UPDATE", (first,)
        )
        assert jobs_b.claim_next().job_id == second


def test_events_are_numbered_per_job_and_can_be_replayed_from_any_point(jobs):
    one = jobs.submit("one", BUDGET)
    two = jobs.submit("two", BUDGET)
    for step in ("planner", "searcher", "screener"):
        jobs.add_event(one, "step", {"step": step})
    jobs.add_event(two, "step", {"step": "planner"})

    assert [e.seq for e in jobs.events_after(one)] == [1, 2, 3, 4]
    assert [e.data for e in jobs.events_after(one, 2)] == [
        {"step": "searcher"},
        {"step": "screener"},
    ]
    assert [e.seq for e in jobs.events_after(two)] == [1, 2]


def test_finish_stores_the_result_and_fail_the_error(jobs):
    done = jobs.submit("done", BUDGET)
    failed = jobs.submit("failed", BUDGET)
    jobs.claim_next()
    jobs.claim_next()

    jobs.finish(done, {"review": "Judges are biased [arXiv:2406.07791]."}, "abc123")
    jobs.fail(failed, "LLMUnavailableError: every provider failed")

    assert jobs.get(done).status == "done"
    assert jobs.get(done).trace_id == "abc123"
    assert jobs.result(done) == {"review": "Judges are biased [arXiv:2406.07791]."}
    assert jobs.get(failed).error == "LLMUnavailableError: every provider failed"
    assert kinds(jobs, failed)[-1][1]["status"] == "failed"
    assert [job.question for job in jobs.recent()] == ["failed", "done"]


def test_jobs_left_running_by_a_dead_worker_are_retried_then_given_up(jobs):
    job_id = jobs.submit("q", BUDGET)
    jobs.claim_next()  # attempt 1; then the worker dies

    assert jobs.recover_orphans(max_attempts=2) == {job_id: "queued"}
    assert jobs.claim_next().attempts == 2  # attempt 2; it dies again

    assert jobs.recover_orphans(max_attempts=2) == {job_id: "failed"}
    job = jobs.get(job_id)
    assert job.status == "failed"
    assert job.error == "the worker stopped during each of 2 attempts"
    assert jobs.recover_orphans() == {}


def test_only_one_connection_can_hold_the_worker_lock(connect):
    key = random.randint(1, 2**31)  # not the real worker's key
    first, second = JobStore(connect()), JobStore(connect())

    assert first.try_lock_worker(key)
    assert not second.try_lock_worker(key)
    first.close()  # the worker dies: Postgres releases its lock
    assert second.try_lock_worker(key)


# --- the pause for review --------------------------------------------------------

REQUEST = {"question": "q", "kept": [{"arxiv_id": "2406.07791"}], "dropped": []}


def paused_job(jobs) -> str:
    job_id = jobs.submit("q", BUDGET, pause_for_review=True)
    jobs.claim_next()
    jobs.pause(job_id, REQUEST)
    return job_id


def test_a_paused_job_waits_with_its_request(jobs):
    job_id = paused_job(jobs)

    job = jobs.get(job_id)
    assert (job.status, job.pause_for_review) == ("awaiting_review", True)
    assert job.paused_at is not None
    assert jobs.review_request(job_id) == REQUEST
    assert jobs.claim_next() is None  # nothing to do until someone decides


def test_a_decision_queues_the_job_again_as_a_new_run(jobs):
    job_id = paused_job(jobs)

    assert jobs.decide(job_id, {"remove": ["2406.07791"], "add": []})

    resumed = jobs.claim_next()
    assert resumed.decision == {"remove": ["2406.07791"], "add": []}
    assert resumed.attempts == 1  # a new run, not a second try
    assert jobs.events_after(job_id)[-1].data == {
        "status": "running",
        "attempt": 1,
        "after": "your review",
    }


def test_only_a_waiting_job_takes_a_decision(jobs):
    job_id = jobs.submit("q", BUDGET)
    assert not jobs.decide(job_id, {"remove": [], "add": []})
    assert jobs.get(job_id).status == "queued"


def test_a_pause_nobody_answers_expires(jobs):
    stale, fresh = paused_job(jobs), paused_job(jobs)
    jobs._conn.execute(
        "UPDATE review_jobs SET paused_at = now() - interval '25 hours' "
        "WHERE job_id = %s",
        (stale,),
    )

    assert jobs.expire_reviews(max_hours=24) == [stale]
    assert jobs.get(stale).status == "expired"
    assert jobs.get(fresh).status == "awaiting_review"


def test_a_failed_job_can_be_retried(jobs):
    job_id = jobs.submit("q", BUDGET)
    jobs.claim_next()
    jobs.fail(job_id, "LLMUnavailableError: every provider failed")

    assert jobs.retry(job_id)
    job = jobs.get(job_id)
    assert (job.status, job.attempts, job.error) == ("queued", 0, None)
    assert not jobs.retry(job_id)  # only failed jobs


def test_labels_are_saved_once_per_paper(jobs):
    job_id = jobs.submit("How biased are LLM judges?", BUDGET)
    edit = {
        "arxiv_id": "2305.17926",
        "action": "removed",
        "label": "screener_false_positive",
        "title": "Large Language Models are not Fair Evaluators",
        "screener_score": 7,
    }

    jobs.save_labels(job_id, "How biased are LLM judges?", [edit])
    jobs.save_labels(job_id, "How biased are LLM judges?", [edit])  # a retried run

    [label] = jobs.labels()
    assert (label["arxiv_id"], label["label"], label["screener_score"]) == (
        "2305.17926",
        "screener_false_positive",
        7,
    )


def test_a_database_from_before_the_pause_is_upgraded(connect):
    conn = connect()
    conn.execute(  # the review_jobs table as 5.2b created it
        """
        CREATE TABLE review_jobs (
            job_id uuid PRIMARY KEY, question text NOT NULL, budget jsonb NOT NULL,
            status text NOT NULL
                CHECK (status IN ('queued', 'running', 'done', 'failed')),
            attempts integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now(), started_at timestamptz,
            heartbeat_at timestamptz, finished_at timestamptz, trace_id text,
            error text, result jsonb
        )
        """
    )
    jobs = JobStore(conn)
    jobs.init_schema()

    assert jobs.get(paused_job(jobs)).status == "awaiting_review"
