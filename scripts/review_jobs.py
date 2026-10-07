"""Submit review jobs and follow them; scripts/review_worker.py runs them.

    uv run --env-file .env python scripts/review_jobs.py submit "How biased are LLM judges?"
    uv run --env-file .env python scripts/review_jobs.py watch <job id>
    uv run --env-file .env python scripts/review_jobs.py list
    uv run --env-file .env python scripts/review_jobs.py show <job id>

watch follows a job's events until it finishes. Ctrl+C only stops watching:
the job keeps running, and watching again replays its events from the start
(or from --after N).
"""

import argparse
import os
import sys
import time
from datetime import UTC, datetime

from arxiv_agent.review.events import describe_event, format_report
from arxiv_agent.review.state import Budget
from arxiv_agent.storage.job_store import JobStore

FINISHED = ("done", "failed")
SILENT_SECONDS = 60  # no heartbeat for this long: the worker probably isn't running


def submit(jobs: JobStore, args: argparse.Namespace) -> None:
    budget = Budget(
        max_llm_calls=args.max_llm_calls,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
    )
    job_id = jobs.submit(args.question, budget.model_dump())
    print(f"queued job {job_id}")
    print(f"follow it with: python scripts/review_jobs.py watch {job_id}")


def print_events(jobs: JobStore, job_id: str, seq: int) -> int:
    for event in jobs.events_after(job_id, seq):
        at = event.at.astimezone()  # stored in UTC, shown in local time
        print(f"[{event.seq:3}] {at:%H:%M:%S} {describe_event(event.kind, event.data)}")
        seq = event.seq
    return seq


def watch(jobs: JobStore, args: argparse.Namespace) -> None:
    if jobs.get(args.job_id) is None:
        sys.exit(f"no job {args.job_id}")
    seq, said = args.after, set()
    while True:
        seq = print_events(jobs, args.job_id, seq)
        job = jobs.get(args.job_id)
        if job.status in FINISHED:
            # The status changes just before its event is written: read once more.
            print_events(jobs, args.job_id, seq)
            print(
                f"\nfinished. See the result: python scripts/review_jobs.py show {job.job_id}"
            )
            return
        if job.status == "queued" and "queued" not in said:
            print("(queued: waiting for the worker)")
            said.add("queued")
        if job.status == "running" and job.heartbeat_at and "silent" not in said:
            silent = (datetime.now(UTC) - job.heartbeat_at).total_seconds()
            if silent > SILENT_SECONDS:
                print(
                    f"(no heartbeat for {silent:.0f} s: is the worker still running?)"
                )
                said.add("silent")
        time.sleep(1)


def list_jobs(jobs: JobStore, args: argparse.Namespace) -> None:
    for job in jobs.recent(args.limit):
        took = ""
        if job.started_at and job.finished_at:
            took = f" in {(job.finished_at - job.started_at).total_seconds():.0f} s"
        print(
            f"{job.job_id}  {job.created_at.astimezone():%m-%d %H:%M}  {job.status:<7}{took}  "
            f"{job.question[:60]}"
        )


def show(jobs: JobStore, args: argparse.Namespace) -> None:
    job = jobs.get(args.job_id)
    if job is None:
        sys.exit(f"no job {args.job_id}")
    print(f"{job.question}\nstatus: {job.status} (attempts: {job.attempts})")
    if job.error:
        print(f"error: {job.error}")
    result = jobs.result(job.job_id)
    if result:
        print("\n" + format_report(result))
    if job.trace_id:
        print(f"\ntrace id: {job.trace_id} (Langfuse session: {job.job_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    budget = Budget()

    p = commands.add_parser("submit", help="queue a review")
    p.add_argument("question")
    p.add_argument("--max-llm-calls", type=int, default=budget.max_llm_calls)
    p.add_argument("--max-tokens", type=int, default=budget.max_tokens)
    p.add_argument("--max-seconds", type=float, default=budget.max_seconds)
    p.set_defaults(handler=submit)

    p = commands.add_parser("watch", help="follow a job's progress")
    p.add_argument("job_id")
    p.add_argument("--after", type=int, default=0, help="skip events up to this number")
    p.set_defaults(handler=watch)

    p = commands.add_parser("list", help="recent jobs")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(handler=list_jobs)

    p = commands.add_parser("show", help="a job's status and result")
    p.add_argument("job_id")
    p.set_defaults(handler=show)

    args = parser.parse_args()
    jobs = JobStore.connect(os.environ["DATABASE_URL"])
    try:
        args.handler(jobs, args)
    except KeyboardInterrupt:
        print("\nstopped watching; the job keeps running")
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
