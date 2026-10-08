"""Repair or remove: run both on the same failing answers and score them.

    uv run --env-file .env python scripts/build_repair_eval.py    (once)
    uv run --env-file .env python scripts/run_repair_eval.py
    uv run --env-file .env python scripts/run_repair_eval.py --limit 10

See arxiv_agent/evals/repair_eval.py for the cases and the scores. Each case
runs the support check once; remove and repair act on the same verdicts.
Results go to data/eval_runs/<run id>/.
"""

import argparse
import contextvars
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from langfuse import get_client, propagate_attributes

from arxiv_agent.evals.judge import JUDGE_MODEL, SCORES, Judge, JudgeError
from arxiv_agent.evals.provenance import git_version
from arxiv_agent.evals.repair_eval import (
    BROKEN_KINDS,
    Outcome,
    RepairCase,
    RepairResult,
    break_gone,
    others_kept,
    summarize,
)
from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL_ID, Embedder
from arxiv_agent.llm import LLMUnavailableError
from arxiv_agent.qa.answerer import REPAIR_PROMPT_VERSION, Answerer
from arxiv_agent.qa.checker import CheckedAnswer, check_answer
from arxiv_agent.qa.support import FAILING, SupportChecker, apply_support
from arxiv_agent.retrieval.retriever import build_retriever
from arxiv_agent.review.critic import CRITIC_PROMPT_VERSION
from arxiv_agent.storage.chunk_store import ChunkStore, SearchHit

CASES = Path("data/evals/repair_cases.jsonl")
RUNS_DIR = Path("data/eval_runs")
SHOWN = 6


class Scorer:
    # Grades an answer against the case's gold answers; the same text is
    # graded once per case (no check, remove and repair often agree).
    def __init__(self, judge: Judge, case: RepairCase) -> None:
        self._judge = judge
        self._case = case
        self._graded: dict[tuple[str, str], float | None] = {}

    def grade(self, status: str, text: str) -> float | None:
        if (status, text) not in self._graded:
            self._graded[status, text] = self._grade(status, text)
        return self._graded[status, text]

    def _grade(self, status: str, text: str) -> float | None:
        if status == "refused":
            return 0.0  # every question here is answerable
        try:
            verdict = self._judge.grade(
                self._case.question, self._case.gold_answers, text
            )
        except JudgeError:
            return None
        return SCORES[verdict.verdict]

    def outcome(self, answer: CheckedAnswer) -> Outcome:
        return Outcome(
            text=answer.text,
            status=answer.status,
            break_gone=break_gone(self._case, answer.text),
            others_kept=others_kept(self._case, answer.text),
            correctness=self.grade(answer.status, answer.text),
        )


def run_case(
    case: RepairCase,
    hits: list[SearchHit],
    answerer: Answerer,
    checker: SupportChecker,
    judge: Judge,
    langfuse,
) -> RepairResult:
    scorer = Scorer(judge, case)
    with langfuse.start_as_current_observation(
        as_type="chain", name="repair-case", input={"id": case.id, "kind": case.kind}
    ) as span:
        broken = check_answer(case.broken, n_sources=len(hits))
        support = checker.check(case.broken, hits)
        failed = [s for s in support if s.verdict in FAILING]
        none = scorer.outcome(broken)
        actions = {"none": none, "remove": none, "repair": none}
        repair, repair_ms = None, None
        if failed:
            actions["remove"] = scorer.outcome(apply_support(broken, support))
            start = time.perf_counter()
            repaired, repair = answerer.repair(case.question, hits, broken, support)
            repair_ms = 1000 * (time.perf_counter() - start)
            actions["repair"] = scorer.outcome(repaired)
        span.update(output={a: o.text for a, o in actions.items()})

    real = case.kind == "real"
    return RepairResult(
        id=case.id,
        kind=case.kind,
        flagged=bool(failed),
        flagged_break=None
        if real
        else any(s.sentence == case.broken_sentence for s in failed),
        false_flags=0
        if real
        else sum(s.sentence != case.broken_sentence for s in failed),
        original=scorer.grade("answered", case.original),
        actions=actions,
        repair_outcome=repair and repair.outcome,
        repair_reason=repair and repair.reason,
        repair_ms=repair_ms,
    )


def fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.2f}"


def print_table(rows: dict, original: float | None) -> None:
    print(
        f"  {'':16}{'break gone':>12}{'correct':>10}{'refused':>10}{'others kept':>13}"
    )
    print(f"  {'unbroken answer':16}{'':>12}{fmt(original):>10}")
    labels = {"none": "no check", "remove": "remove (6.3b)", "repair": "repair (6.3c)"}
    for action, s in rows.items():
        print(
            f"  {labels[action]:16}{fmt(s['break_gone']):>12}{fmt(s['correctness']):>10}"
            f"{fmt(s['refused']):>10}{fmt(s['others_kept']):>13}"
        )


def print_report(summary: dict, results: list[RepairResult], cases: dict) -> None:
    b = summary["broken"]
    print(
        f"\n{b['cases']} broken answers: the check failed the broken sentence in "
        f"{fmt(b['flagged_break'])}; {b['false_flags']} supported sentences "
        f"failed alongside"
    )
    print_table(b["all"], b["original_correctness"])
    flagged = [r for r in results if r.kind != "real" and r.flagged and not r.error]
    print(f"\n where the check fired ({len(flagged)}):")
    print_table(b["flagged"], None)
    print("\n by kind (break gone / correct):  remove | repair")
    for kind in BROKEN_KINDS:
        k = b["by_kind"][kind]
        print(
            f"  {kind:16} {fmt(k['remove']['break_gone'])} / "
            f"{fmt(k['remove']['correctness'])} | {fmt(k['repair']['break_gone'])}"
            f" / {fmt(k['repair']['correctness'])}"
        )
    r = summary["real"]
    if r["cases"]:
        a = r["all"]
        print(
            f"\n{r['cases']} real failures: correct unbroken {fmt(r['original_correctness'])}"
            f" | remove {fmt(a['remove']['correctness'])}"
            f" | repair {fmt(a['repair']['correctness'])}"
        )
    p = summary["repair"]
    ms = "-" if p["ms_p50"] is None else f"{p['ms_p50']:.0f} ms"
    print(f"\nRepair used {p['used']} times, {p['fallbacks']} fell back; p50 {ms}")

    worse = [
        x
        for x in results
        if x.repair_outcome
        and (
            x.actions["repair"].break_gone is False
            or (x.actions["repair"].correctness or 0)
            < (x.actions["remove"].correctness or 0)
        )
    ]
    if worse:
        print(
            f"\nRepair kept the break or did worse than removing ({len(worse)}), e.g."
        )
    for x in worse[:SHOWN]:
        print(f"  [{x.kind}] {cases[x.id].broken_sentence[:120]}")
        print(f"      repair: {x.actions['repair'].text[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--providers", default="deepinfra,nscale")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
    langfuse = get_client()
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}-repair"
    lines = args.cases.read_text(encoding="utf-8").splitlines()
    cases = [RepairCase.model_validate_json(line) for line in lines if line.strip()]
    cases = cases[: args.limit] if args.limit else cases

    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    chunks = {}
    for paper in {(c.arxiv_id, c.version) for c in cases}:
        chunks.update({chunk.chunk_id: chunk for chunk in store.get_chunks(*paper)})
    hits = {
        c.id: [SearchHit(chunk=chunks[cid], score=0.0) for cid in c.chunk_ids]
        for c in cases
    }
    checker = SupportChecker(langfuse=langfuse)
    answerer = Answerer(
        build_retriever(
            "rerank", store, Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
        ),
        model=args.model,
        providers=args.providers.split(","),
        langfuse=langfuse,
        support=checker,
    )
    judge = Judge(langfuse=langfuse)

    def one(case: RepairCase) -> RepairResult:
        try:
            result = run_case(case, hits[case.id], answerer, checker, judge, langfuse)
        except LLMUnavailableError as exc:
            result = RepairResult(
                id=case.id,
                kind=case.kind,
                flagged=False,
                flagged_break=None,
                false_flags=0,
                original=None,
                actions={},
                error=f"{exc}"[:200],
            )
        print(f"  {case.id[:40]:40} {result.repair_outcome or '-'}", flush=True)
        return result

    start = time.perf_counter()
    with (
        propagate_attributes(session_id=run_id),
        ThreadPoolExecutor(args.concurrency) as pool,
    ):
        futures = [pool.submit(contextvars.copy_context().run, one, c) for c in cases]
        results = [future.result() for future in futures]
    store.close()
    langfuse.flush()

    summary = summarize(results)
    out = args.out / run_id
    out.mkdir(parents=True)
    (out / "results.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8"
    )
    meta = {
        "run_id": run_id,
        "git": git_version(),
        "duration_s": time.perf_counter() - start,
        "model": args.model,
        "judge_model": JUDGE_MODEL,
        "repair_prompt_version": REPAIR_PROMPT_VERSION,
        "critic_prompt_version": CRITIC_PROMPT_VERSION,
    }
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(f"\nRun {run_id} | {len(cases)} cases in {meta['duration_s']:.0f} s")
    print_report(summary, results, {c.id: c for c in cases})
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
