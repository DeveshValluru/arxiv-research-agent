"""Build the repair eval cases from real answers.

    uv run --env-file .env python scripts/build_repair_eval.py

Asks the Answerer (support check on, repair off) every answerable question in
the Q&A eval, then breaks one supported sentence of each answer in each of
three ways (see arxiv_agent/evals/repair_eval.py). Answers the check fails on
its own become "real" cases. The answers paraphrase paper text, so the cases
go to data/evals/repair_cases.jsonl, not to git. Answers vary between runs, so
build once and run the eval on the saved cases.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

from langfuse import get_client, propagate_attributes

from arxiv_agent.evals.repair_eval import RepairCase, break_answer
from arxiv_agent.evals.runner import load_items
from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL_ID, Embedder
from arxiv_agent.llm import LLMUnavailableError
from arxiv_agent.qa.answerer import Answerer, NotIndexedError
from arxiv_agent.qa.support import FAILING, SupportChecker
from arxiv_agent.retrieval.retriever import build_retriever
from arxiv_agent.storage.chunk_store import ChunkStore

EVAL_SETS = [Path("evals/qa_qasper.jsonl"), Path("evals/qa_survey.jsonl")]
OUTPUT = Path("data/evals/repair_cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--providers", default="deepinfra,nscale")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
    langfuse = get_client()
    items = [i for i in load_items(EVAL_SETS) if i.type == "answerable"]
    items = items[: args.limit]
    store = ChunkStore.connect(os.environ["DATABASE_URL"])
    embedder = Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
    answerer = Answerer(
        build_retriever("rerank", store, embedder),
        model=args.model,
        providers=args.providers.split(","),
        langfuse=langfuse,
        support=SupportChecker(langfuse=langfuse),
        repair=False,
    )

    cases: list[RepairCase] = []
    for n, item in enumerate(items, start=1):
        try:
            with propagate_attributes(session_id="repair-eval-build"):
                result = answerer.ask(item.question, item.arxiv_id, item.version)
        except (LLMUnavailableError, NotIndexedError) as exc:
            print(f"[{n:2}/{len(items)}] {item.id[:24]} skipped: {exc}", flush=True)
            continue
        if not result.support:  # refused or invalid: nothing to break
            print(f"[{n:2}/{len(items)}] {item.id[:24]} {result.answer.status}")
            continue
        # The answer as written, before the check removed anything.
        original = " ".join(s.sentence for s in result.support)
        common = {
            "item_id": item.id,
            "question": item.question,
            "gold_answers": item.gold_answers,
            "arxiv_id": item.arxiv_id,
            "version": item.version,
            "chunk_ids": [source.chunk_id for source in result.sources],
            "original": original,
        }
        if any(s.verdict in FAILING for s in result.support):
            made = [("real", original, "", "", None)]
        else:
            made = break_answer(original, result.support)
        for kind, broken, sentence, broken_sentence, marker in made:
            cases.append(
                RepairCase(
                    id=f"{item.id}:{kind}",
                    kind=kind,
                    broken=broken,
                    sentence=sentence,
                    broken_sentence=broken_sentence,
                    marker=marker,
                    **common,
                )
            )
        print(
            f"[{n:2}/{len(items)}] {item.id[:24]} "
            f"{', '.join(kind for kind, *_ in made) or 'nothing to break'}",
            flush=True,
        )
    store.close()
    langfuse.flush()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    kinds = Counter(case.kind for case in cases)
    print(f"\n{len(cases)} cases {dict(kinds)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
