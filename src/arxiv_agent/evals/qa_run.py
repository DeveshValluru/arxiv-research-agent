"""Running the Q&A eval: shared by scripts/run_eval.py and scripts/eval_gate.py.

A question can run several times (repeats: the gate needs them to see past
run-to-run noise) and several questions at once (concurrency: CI has minutes,
not hours). Every run is its own trace, all in one Langfuse session, with its
scores attached.
"""

import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langfuse import Langfuse, propagate_attributes

from arxiv_agent.evals.judge import Judge
from arxiv_agent.evals.runner import EvalItem, ItemScore, score_item
from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, DEFAULT_MODEL_ID, Embedder
from arxiv_agent.llm import LLMUnavailableError
from arxiv_agent.qa.answerer import Answerer, NotIndexedError
from arxiv_agent.qa.support import SupportChecker
from arxiv_agent.retrieval.retriever import build_retriever
from arxiv_agent.storage.chunk_store import ChunkStore

EVALS_DIR = Path("evals")
MODEL = "Qwen/Qwen3-32B"
PROVIDERS = "deepinfra,nscale"


def eval_sets(directory: Path = EVALS_DIR) -> list[Path]:
    # Every Q&A eval set, including the questions the flywheel adds (7.3).
    return sorted(directory.glob("qa_*.jsonl"))


def build_answerer(
    store: ChunkStore,
    langfuse: Langfuse,
    *,
    model: str = MODEL,
    providers: list[str] | None = None,
    k: int = 5,
    retriever: str = "rerank",
    support: bool = True,
    repair: bool = True,
) -> Answerer:
    embedder = Embedder(DEFAULT_MODEL_ID, query_prefix=BGE_QUERY_PREFIX)
    return Answerer(
        build_retriever(retriever, store, embedder),
        model=model,
        providers=providers or PROVIDERS.split(","),
        k=k,
        langfuse=langfuse,
        support=SupportChecker(langfuse=langfuse) if support else None,
        repair=repair,
    )


def attach_scores(langfuse: Langfuse, score: ItemScore, run_id: str) -> None:
    if score.trace_id is None:
        return
    numeric = {"correctness": score.correctness, "token_f1": score.f1}
    boolean = {"evidence_hit": score.evidence_hit, "refusal_ok": score.refusal_ok}
    for name, value in numeric.items():
        if value is not None:
            langfuse.create_score(
                name=name,
                value=value,
                trace_id=score.trace_id,
                comment=score.judge_reasoning if name == "correctness" else None,
                metadata={"run_id": run_id},
            )
    for name, value in boolean.items():
        if value is not None:
            langfuse.create_score(
                name=name,
                value=1.0 if value else 0.0,
                data_type="BOOLEAN",
                trace_id=score.trace_id,
                metadata={"run_id": run_id},
            )


def run_questions(
    items: list[EvalItem],
    answerer: Answerer,
    judge: Judge,
    store: ChunkStore,
    *,
    run_id: str,
    langfuse: Langfuse,
    repeats: int = 1,
    concurrency: int = 1,
    progress: Callable[[ItemScore], None] | None = None,
) -> list[ItemScore]:
    # Scores come back in order: question by question, repeats together.
    papers = {(item.arxiv_id, item.version) for item in items}
    chunks = {paper: store.get_chunks(*paper) for paper in papers}

    def one(item: EvalItem, repeat: int) -> ItemScore:
        paper = (item.arxiv_id, item.version)
        try:
            # One Langfuse session per run groups all of its traces together.
            with propagate_attributes(session_id=run_id):
                result = answerer.ask(item.question, *paper)
        except (LLMUnavailableError, NotIndexedError) as exc:
            # One failed question must not end the run: record it and move on.
            score = ItemScore(
                id=item.id,
                source=item.source,
                type=item.type,
                repeat=repeat,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            score = score_item(item, result, chunks[paper], judge)
            score = score.model_copy(update={"repeat": repeat})
            attach_scores(langfuse, score, run_id)
        if progress is not None:
            progress(score)
        return score

    jobs = [(item, repeat) for item in items for repeat in range(repeats)]
    if concurrency == 1:
        return [one(item, repeat) for item, repeat in jobs]
    # Each thread runs in a copy of this context, so its trace stands alone.
    with ThreadPoolExecutor(concurrency) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, one, item, repeat)
            for item, repeat in jobs
        ]
        return [future.result() for future in futures]
