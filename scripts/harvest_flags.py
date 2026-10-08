"""Collect what went wrong in real use as eval candidates (7.3).

    uv run --env-file .env python scripts/harvest_flags.py
    uv run --env-file .env python scripts/harvest_flags.py --since 2026-10-01

Three sources (see arxiv_agent/evals/flywheel.py): Q&A traces a guard flagged
(Langfuse, environment "development" by default: real use, not evals),
sentences the Critic removed from finished reviews, and reviewer edits
(Postgres). New candidates are added to data/evals/flag_candidates.jsonl as
pending; ones already there keep their status. Then run triage_flags.py.
"""

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from langfuse import get_client
from psycopg.rows import dict_row

from arxiv_agent.evals.flywheel import (
    CANDIDATES,
    Candidate,
    critic_candidates,
    label_candidate,
    load_candidates,
    merge,
    qa_candidate,
    save_candidates,
)
from arxiv_agent.qa.support import FAILING


def observations(langfuse, **filters) -> list[dict]:
    # Langfuse's v2 observations API, every page (the legacy trace API is
    # closed to this project).
    found, cursor = [], None
    while True:
        page = langfuse.api.observations.get_many(limit=100, cursor=cursor, **filters)
        found += [o if isinstance(o, dict) else o.dict() for o in page.data]
        cursor = getattr(page.meta, "cursor", None) if page.meta else None
        if not cursor or not page.data:
            return found


def _json(raw):
    # The v2 API returns input and output as raw strings.
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def flagged_qa(langfuse, environment: str, since: datetime) -> list[Candidate]:
    trace_ids = {
        o["traceId"]
        for name in ("ask", "support-check")
        for o in observations(
            langfuse,
            name=name,
            level="WARNING",
            environment=environment,
            fields="core,basic",
            from_start_time=since,
        )
    }
    found = []
    for trace_id in sorted(trace_ids):
        spans = observations(
            langfuse, trace_id=trace_id, fields="core,basic,io", from_start_time=since
        )
        root = next((s for s in spans if s["name"] == "ask"), None)
        check = next((s for s in spans if s["name"] == "support-check"), None)
        if root is None:
            continue
        asked = _json(root["input"])
        verdicts = _json(check["output"]) if check else []
        if isinstance(verdicts, dict):  # since 6.3c: verdicts, repair, answer
            verdicts = verdicts.get("verdicts", [])
        candidate = qa_candidate(
            {
                "trace_id": trace_id,
                "time": str(root["startTime"]),
                "question": asked["question"],
                "paper": asked["paper"],
                "answer": _json(root["output"]),
                "status": "invalid" if root.get("level") == "WARNING" else "answered",
                "problems": (root.get("statusMessage") or "").split("; "),
                "failed": [v["sentence"] for v in verdicts if v["verdict"] in FAILING],
            }
        )
        if candidate is not None:
            found.append(candidate)
    return found


def from_postgres(url: str, since: datetime) -> list[Candidate]:
    with psycopg.connect(url, row_factory=dict_row) as conn:
        jobs = conn.execute(
            "SELECT job_id::text AS job_id, created_at, trace_id, result "
            "FROM review_jobs WHERE status = 'done' AND created_at >= %s "
            "AND jsonb_typeof(result->'removed') = 'array' "
            "AND jsonb_array_length(result->'removed') > 0",
            (since,),
        ).fetchall()
        labels = conn.execute(
            "SELECT l.job_id::text AS job_id, l.arxiv_id, l.label, l.question, "
            "l.title, j.created_at FROM review_labels l JOIN review_jobs j "
            "USING (job_id) WHERE j.created_at >= %s",
            (since,),
        ).fetchall()
    found = [
        candidate
        for job in jobs
        for candidate in critic_candidates(
            job["job_id"], str(job["created_at"]), job["trace_id"], job["result"]
        )
    ]
    return found + [label_candidate(row) for row in labels]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", default="2026-01-01", help="YYYY-MM-DD")
    parser.add_argument(
        "--environment",
        default="development",
        help="Langfuse environment to read Q&A flags from (real use, not evals)",
    )
    parser.add_argument("--out", type=Path, default=CANDIDATES)
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
    found = flagged_qa(get_client(), args.environment, since)
    found += from_postgres(os.environ["DATABASE_URL"], since)
    existing = load_candidates(args.out)
    merged = merge(existing, found)
    save_candidates(merged, args.out)

    new = merged[len(existing) :]
    pending = Counter(c.kind for c in merged if c.status == "pending")
    print(
        f"found {len(found)} flags, {len(new)} new "
        f"({dict(Counter(c.kind for c in new))}); pending: {dict(pending)}\n"
        f"wrote {args.out}; next: scripts/triage_flags.py"
    )


if __name__ == "__main__":
    main()
