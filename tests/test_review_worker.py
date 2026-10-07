import asyncio

from arxiv_agent.llm import Usage
from arxiv_agent.review.graph import ReviewRun
from arxiv_agent.review.state import Budget
from arxiv_agent.review.worker import ReviewWorker

BUDGET = Budget(max_seconds=120).model_dump()
REQUEST = {"question": "q", "kept": [{"arxiv_id": "2406.07791"}], "dropped": []}
EDIT = {
    "arxiv_id": "2305.17926",
    "action": "removed",
    "label": "screener_false_positive",
    "title": "Large Language Models are not Fair Evaluators",
    "screener_score": 7,
}


class FakeRunner:
    # Stands in for run_review: emits step events, then finishes (or pauses
    # for review, raises, or takes a while).
    def __init__(self, pause=None, error=None, seconds=0.0, edits=()):
        self.pause = pause
        self.error = error
        self.seconds = seconds
        self.edits = list(edits)
        self.jobs = []

    async def __call__(self, job, on_event):
        self.jobs.append(job)
        on_event("step", {"step": "planner", "searches": ["a", "b"]})
        if self.edits:
            on_event("step", {"step": "human_review", "edits": self.edits})
        await asyncio.sleep(self.seconds)
        if self.error:
            raise self.error
        if self.pause:
            return ReviewRun({"question": job.question}, "trace-1", self.pause)
        on_event("step", {"step": "finalize", "removed": 0})
        state = {
            "question": job.question,
            "review": "Judges are biased.",
            "spent": Usage(),
        }
        return ReviewRun(state, "trace-2", None)


class Forgetful:
    def __init__(self):
        self.forgotten = []

    async def __call__(self, job_id):
        self.forgotten.append(job_id)


def worker(jobs, runner, forget=None, **settings):
    return ReviewWorker(jobs, runner, forget=forget or Forgetful(), **settings)


def run_next(jobs, runner, **settings) -> bool:
    return asyncio.run(worker(jobs, runner, **settings).run_next())


def statuses(jobs, job_id):
    return [
        e.data.get("status") or e.data.get("step") for e in jobs.events_after(job_id)
    ]


def test_the_worker_runs_a_job_and_stores_its_events_and_result(jobs):
    job_id = jobs.submit("How biased are LLM judges?", BUDGET)
    runner, forget = FakeRunner(), Forgetful()

    assert run_next(jobs, runner, forget=forget)

    assert [job.job_id for job in runner.jobs] == [job_id]
    job = jobs.get(job_id)
    assert (job.status, job.trace_id) == ("done", "trace-2")
    assert jobs.result(job_id)["review"] == "Judges are biased."
    assert statuses(jobs, job_id) == [
        "queued",
        "running",
        "planner",
        "finalize",
        "done",
    ]
    assert forget.forgotten == [job_id]  # done for good: its checkpoints go


def test_a_job_that_pauses_waits_for_review_and_keeps_its_checkpoints(jobs):
    job_id = jobs.submit("q", BUDGET, pause_for_review=True)
    forget = Forgetful()

    run_next(jobs, FakeRunner(pause=REQUEST), forget=forget)

    assert jobs.get(job_id).status == "awaiting_review"
    assert jobs.review_request(job_id) == REQUEST
    assert statuses(jobs, job_id)[-1] == "awaiting_review"
    assert forget.forgotten == []


def test_the_decision_reaches_the_runner_and_the_edits_become_labels(jobs):
    job_id = jobs.submit("How biased are LLM judges?", BUDGET, pause_for_review=True)
    run_next(jobs, FakeRunner(pause=REQUEST))
    jobs.decide(job_id, {"remove": ["2305.17926"], "add": []})
    runner = FakeRunner(edits=[EDIT])

    run_next(jobs, runner)

    assert runner.jobs[0].decision == {"remove": ["2305.17926"], "add": []}
    assert jobs.get(job_id).status == "done"
    [label] = jobs.labels()
    assert (label["arxiv_id"], label["label"]) == (
        "2305.17926",
        "screener_false_positive",
    )


def test_edits_are_labelled_even_if_the_review_then_fails(jobs):
    jobs.submit("q", BUDGET)
    run_next(jobs, FakeRunner(edits=[EDIT], error=RuntimeError("reader exploded")))
    assert [label["arxiv_id"] for label in jobs.labels()] == ["2305.17926"]


def test_a_failing_job_is_recorded_and_the_worker_moves_on(jobs):
    bad = jobs.submit("bad", BUDGET)
    good = jobs.submit("good", BUDGET)
    forget = Forgetful()

    assert run_next(
        jobs, FakeRunner(error=RuntimeError("provider exploded")), forget=forget
    )
    assert run_next(jobs, FakeRunner(), forget=forget)

    assert jobs.get(bad).status == "failed"
    assert jobs.get(bad).error == "RuntimeError: provider exploded"
    assert statuses(jobs, bad)[2] == "planner"  # you can see where it broke
    assert jobs.get(good).status == "done"
    assert forget.forgotten == [good]  # the failed one can still be retried


def test_an_empty_queue_means_nothing_to_do(jobs):
    runner = FakeRunner()
    assert not run_next(jobs, runner)
    assert runner.jobs == []


def test_a_long_job_keeps_its_heartbeat_fresh(jobs):
    job_id = jobs.submit("slow", BUDGET)

    run_next(jobs, FakeRunner(seconds=0.3), heartbeat_seconds=0.05)

    job = jobs.get(job_id)
    assert job.heartbeat_at > job.started_at


def test_the_sweep_expires_old_pauses_and_forgets_them(jobs):
    job_id = jobs.submit("q", BUDGET, pause_for_review=True)
    run_next(jobs, FakeRunner(pause=REQUEST))
    jobs._conn.execute(
        "UPDATE review_jobs SET paused_at = now() - interval '2 days' WHERE job_id = %s",
        (job_id,),
    )
    forget = Forgetful()

    asyncio.run(worker(jobs, FakeRunner(), forget=forget).sweep())

    assert jobs.get(job_id).status == "expired"
    assert forget.forgotten == [job_id]
