import asyncio

from arxiv_agent.llm import Usage
from arxiv_agent.review.state import Budget
from arxiv_agent.review.worker import ReviewWorker

BUDGET = Budget(max_seconds=120)


class FakeRunner:
    # Stands in for run_review: emits two step events, then returns a final
    # state (or raises, or takes a while).
    def __init__(self, error: Exception | None = None, seconds: float = 0.0):
        self.error = error
        self.seconds = seconds
        self.calls: list[tuple[str, Budget, str]] = []

    async def __call__(self, question, budget, on_event, job_id):
        self.calls.append((question, budget, job_id))
        on_event("step", {"step": "planner", "searches": ["a", "b"]})
        await asyncio.sleep(self.seconds)
        if self.error:
            raise self.error
        on_event("step", {"step": "finalize", "removed": 0})
        state = {"question": question, "review": "Judges are biased.", "spent": Usage()}
        return state, "trace-123"


def run_next(jobs, runner, **settings) -> bool:
    return asyncio.run(ReviewWorker(jobs, runner, **settings).run_next())


def test_the_worker_runs_a_job_and_stores_its_events_and_result(jobs):
    job_id = jobs.submit("How biased are LLM judges?", BUDGET.model_dump())
    runner = FakeRunner()

    assert run_next(jobs, runner)

    assert runner.calls == [("How biased are LLM judges?", BUDGET, job_id)]
    job = jobs.get(job_id)
    assert (job.status, job.trace_id) == ("done", "trace-123")
    assert jobs.result(job_id)["review"] == "Judges are biased."
    assert [
        (e.kind, e.data.get("status") or e.data.get("step"))
        for e in jobs.events_after(job_id)
    ] == [
        ("status", "queued"),
        ("status", "running"),
        ("step", "planner"),
        ("step", "finalize"),
        ("status", "done"),
    ]


def test_a_failing_job_is_recorded_and_the_worker_moves_on(jobs):
    bad = jobs.submit("bad", BUDGET.model_dump())
    good = jobs.submit("good", BUDGET.model_dump())

    assert run_next(jobs, FakeRunner(error=RuntimeError("provider exploded")))
    assert run_next(jobs, FakeRunner())

    assert jobs.get(bad).status == "failed"
    assert jobs.get(bad).error == "RuntimeError: provider exploded"
    # The events up to the failure are kept: you can see where it broke.
    assert [e.data.get("step") for e in jobs.events_after(bad)][2] == "planner"
    assert jobs.get(good).status == "done"


def test_an_empty_queue_means_nothing_to_do(jobs):
    runner = FakeRunner()
    assert not run_next(jobs, runner)
    assert runner.calls == []


def test_a_long_job_keeps_its_heartbeat_fresh(jobs):
    job_id = jobs.submit("slow", BUDGET.model_dump())

    run_next(jobs, FakeRunner(seconds=0.3), heartbeat_seconds=0.05)

    job = jobs.get(job_id)
    assert job.heartbeat_at > job.started_at
