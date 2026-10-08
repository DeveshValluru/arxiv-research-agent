import time
from collections.abc import Callable
from typing import Literal

from huggingface_hub import InferenceClient
from langfuse import Langfuse, get_client, propagate_attributes
from pydantic import BaseModel, ConfigDict

from arxiv_agent.guardrails.output import OutputGuard
from arxiv_agent.llm import ANSWER_TIMEOUT, LLMUnavailableError, chat_with_failover
from arxiv_agent.qa.checker import REFUSAL, CheckedAnswer, check_answer
from arxiv_agent.qa.support import (
    FAILING,
    SentenceSupport,
    SupportChecker,
    apply_support,
)
from arxiv_agent.retrieval.retriever import Retriever
from arxiv_agent.storage.chunk_store import SearchHit

# Bump PROMPT_VERSION whenever SYSTEM_PROMPT or the message layout changes, so
# eval scores can say which prompt produced them.
PROMPT_VERSION = 1
SYSTEM_PROMPT = f"""You answer questions about one research paper using ONLY the numbered sources below.

1. Use only the sources. Do not use your own knowledge or other papers, even if you know the answer.
2. After every sentence that states a fact, cite its source(s) in square brackets, like [S2] or [S1][S3]. Cite only sources that appear below.
3. If the sources do not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL}
4. If the sources answer only part of the question, answer that part and say which part the paper doesn't cover.
5. Be concise: at most 5 sentences.
/no_think"""

# The repair turn (6.3c): sent after the answer when the support check fails
# sentences. On answers with a sentence broken on purpose, where the check
# fired, correctness was 0.31 removing and 0.93 repairing, with every break
# gone either way (scripts/run_repair_eval.py). Bump REPAIR_PROMPT_VERSION
# whenever it changes.
REPAIR_PROMPT_VERSION = 1
REPAIR_REQUEST = """A checker compared each sentence of your answer with the sources it cites. These sentences say more than their sources do:

{flagged}

Rewrite your answer. Keep every other sentence word for word. Fix each sentence above so it says what its cited sources say: correct a detail the sources state differently, drop a detail they don't mention, or leave the sentence out if the sources support none of it. The same rules apply as before. Reply with the answer only.
/no_think"""

NO_TRACE_ID = "0" * 32  # what a disabled Langfuse client reports


class NotIndexedError(LookupError):
    # The paper has no chunks yet. Its own class, so callers that skip such
    # questions don't also swallow IndexError and KeyError (both LookupErrors)
    # from real bugs.
    pass


def repair_request(failed: list[SentenceSupport]) -> str:
    flagged = "\n".join(
        f'{n}. "{s.sentence}"\n   Problem: {s.reason}'
        for n, s in enumerate(failed, start=1)
    )
    return REPAIR_REQUEST.format(flagged=flagged)


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    chunk_id: str
    section_path: list[str]
    score: float
    text: str


class Repair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # repaired: the rewrite was used (after removing what still failed);
    # fallback: it was unusable, so the failing sentences were removed instead
    outcome: Literal["repaired", "fallback"]
    text: str  # the rewrite as the model wrote it, before the re-check
    support: list[SentenceSupport]  # the re-check; unchanged sentences keep theirs
    reason: str | None = None  # why it fell back
    prompt_tokens: int = 0
    completion_tokens: int = 0


class QAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    arxiv_id: str
    version: int
    answer: CheckedAnswer
    sources: list[Source]
    model: str
    provider: str  # the provider that actually answered, after any failover
    llm_attempts: int
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    retrieval_ms: float
    generation_ms: float
    support_ms: float = 0.0  # the claim-support check and any repair
    support: list[SentenceSupport] = []  # its verdict on each sentence
    repair: Repair | None = None  # when sentences failed and repair is on
    prompt_version: int
    retriever: str  # e.g. "hybrid(BAAI/bge-small-en-v1.5+bm25)+rerank(...)"
    chunker_version: int
    trace_id: str | None


def format_sources(hits: list[SearchHit]) -> str:
    # The section path lets the model say where a claim comes from, and lets
    # you check it in the paper afterwards.
    return "\n\n".join(
        f"[S{i}] ({' > '.join(hit.chunk.section_path)}) {hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: list[SearchHit]) -> list[dict]:
    # Rules in the system message (the same on every call); the sources and then
    # the question in the user message, question last.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}",
        },
    ]


class Answerer:
    # Tracing lives here, in the library, so every caller (ask.py, the eval, the
    # API) gets the same trace. Callers only decide whether and where to send it.
    def __init__(
        self,
        retriever: Retriever,
        model: str,
        providers: list[str],
        k: int = 5,
        max_tokens: int = 500,
        clients: dict[str, InferenceClient] | None = None,
        langfuse: Langfuse | None = None,
        sleep: Callable[[float], None] = time.sleep,
        support: SupportChecker | None = None,
        repair: bool = True,
    ) -> None:
        # support: checks each answered sentence against its sources (6.3b);
        # None skips it (tests, and measuring what it changes).
        # repair: rewrite failing sentences once before removing them (6.3c);
        # False only removes them.
        if not providers:
            raise ValueError("need at least one provider")
        self._retriever = retriever
        self._model = model
        self._providers = providers
        self._k = k
        self._max_tokens = max_tokens
        self._clients = clients or {
            p: InferenceClient(provider=p, timeout=ANSWER_TIMEOUT) for p in providers
        }
        self._langfuse = langfuse or get_client()
        self._sleep = sleep
        self._guard = OutputGuard([SYSTEM_PROMPT])
        self._support = support
        self._repair = repair

    @property
    def retriever_name(self) -> str:
        return self._retriever.name

    def ask(self, question: str, arxiv_id: str, version: int) -> QAResult:
        paper = f"{arxiv_id}v{version}"
        with (
            self._langfuse.start_as_current_observation(
                as_type="chain",
                name="ask",
                input={"question": question, "paper": paper},
            ) as root,
            propagate_attributes(
                trace_name="ask",
                tags=[paper, self._model],
                metadata={
                    "prompt_version": str(PROMPT_VERSION),
                    "retriever": self._retriever.name,
                },
            ),
        ):
            hits, retrieval_ms = self._retrieve(question, arxiv_id, version)
            response, provider, attempts, generation_ms = self._generate(
                build_messages(question, hits)
            )

            choice = response.choices[0]
            answer = check_answer(
                choice.message.content or "", n_sources=len(hits), guard=self._guard
            )
            if choice.finish_reason == "length":
                answer = answer.model_copy(
                    update={
                        "status": "invalid",
                        "problems": [
                            *answer.problems,
                            "finish_reason was 'length': the answer was cut off",
                        ],
                    }
                )
            support, repair, support_ms = [], None, 0.0
            if self._support is not None and answer.status == "answered":
                answer, support, repair, support_ms = self._check_support(
                    question, hits, answer
                )

            # Invalid answers become warnings, so they stand out in the trace list.
            root.update(
                output=answer.text,
                level="WARNING" if answer.status == "invalid" else None,
                status_message="; ".join(answer.problems) or None,
                metadata={
                    "status": answer.status,
                    "provider": provider,
                    "chunker_version": hits[0].chunk.chunker_version,
                },
            )
            trace_id = root.trace_id

        return QAResult(
            question=question,
            arxiv_id=arxiv_id,
            version=version,
            answer=answer,
            sources=[
                Source(
                    label=f"S{i}",
                    chunk_id=hit.chunk.chunk_id,
                    section_path=hit.chunk.section_path,
                    score=hit.score,
                    text=hit.chunk.text,
                )
                for i, hit in enumerate(hits, start=1)
            ],
            model=self._model,
            provider=provider,
            llm_attempts=attempts,
            finish_reason=choice.finish_reason,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=getattr(response.usage, "estimated_cost", None),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            support_ms=support_ms,
            support=support,
            repair=repair,
            prompt_version=PROMPT_VERSION,
            retriever=self._retriever.name,
            chunker_version=hits[0].chunk.chunker_version,
            trace_id=None if trace_id == NO_TRACE_ID else trace_id,
        )

    def _check_support(
        self, question: str, hits: list[SearchHit], answer: CheckedAnswer
    ) -> tuple[CheckedAnswer, list[SentenceSupport], Repair | None, float]:
        start = time.perf_counter()
        with self._langfuse.start_as_current_observation(
            as_type="chain", name="support-check", input=answer.text
        ) as span:
            support = self._support.check(answer.text, hits)
            failed = any(s.verdict in FAILING for s in support)
            repair = None
            if failed and self._repair:
                checked, repair = self.repair(question, hits, answer, support)
            else:
                checked = apply_support(answer, support)
            span.update(
                output={
                    "verdicts": [s.model_dump() for s in support],
                    "repair": repair and repair.outcome,
                    "answer": checked.text,
                },
                level="WARNING" if failed else None,
            )
        return checked, support, repair, 1000 * (time.perf_counter() - start)

    def repair(
        self,
        question: str,
        hits: list[SearchHit],
        answer: CheckedAnswer,
        support: list[SentenceSupport],
    ) -> tuple[CheckedAnswer, Repair]:
        # One rewrite, told which sentences failed and why. The rewrite is
        # checked like any answer, and whatever still fails is removed. If it's
        # unusable (no provider, broken format, a refusal, nothing supported
        # left), the failing sentences are removed from the first answer: never
        # worse than not repairing.
        if self._support is None:
            raise ValueError("repair needs a support checker")
        failed = [s for s in support if s.verdict in FAILING]
        removed = apply_support(answer, support)

        def fallback(
            text: str, reason: str, **usage: int
        ) -> tuple[CheckedAnswer, Repair]:
            return removed, Repair(
                outcome="fallback", text=text, support=[], reason=reason, **usage
            )

        messages = [
            *build_messages(question, hits),
            {"role": "assistant", "content": answer.text},
            {"role": "user", "content": repair_request(failed)},
        ]
        try:
            response, _, _, _ = self._generate(messages, name="repair")
        except LLMUnavailableError as exc:
            return fallback("", f"no provider: {exc}"[:200])
        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        rewrite = check_answer(
            choice.message.content or "", n_sources=len(hits), guard=self._guard
        )
        if choice.finish_reason == "length" or rewrite.status != "answered":
            problems = "; ".join(rewrite.problems) or rewrite.status
            return fallback(rewrite.text, f"rewrite unusable: {problems}", **usage)

        resupport = self._support.check(rewrite.text, hits, known=support)
        repaired = apply_support(rewrite, resupport)
        if repaired.status != "answered":
            return removed, Repair(
                outcome="fallback",
                text=rewrite.text,
                support=resupport,  # kept: it shows why the rewrite failed
                reason="nothing supported left",
                **usage,
            )
        notes = [f"repaired ({s.verdict}: {s.reason}): {s.sentence}" for s in failed]
        return repaired.model_copy(
            update={"problems": [*notes, *repaired.problems]}
        ), Repair(outcome="repaired", text=rewrite.text, support=resupport, **usage)

    def _retrieve(
        self, question: str, arxiv_id: str, version: int
    ) -> tuple[list[SearchHit], float]:
        with self._langfuse.start_as_current_observation(
            as_type="retriever",
            name="retrieve",
            input={"question": question, "k": self._k},
            metadata={"retriever": self._retriever.name},
        ) as span:
            start = time.perf_counter()
            hits = self._retriever.retrieve(question, (arxiv_id, version), self._k)
            retrieval_ms = 1000 * (time.perf_counter() - start)
            if not hits:
                raise NotIndexedError(
                    f"{arxiv_id}v{version} has no indexed chunks for "
                    f"{self._retriever.name}; ingest it first"
                )
            span.update(
                output=[
                    {
                        "label": f"S{i}",
                        "chunk_id": hit.chunk.chunk_id,
                        "score": round(hit.score, 4),
                        "section": " > ".join(hit.chunk.section_path),
                    }
                    for i, hit in enumerate(hits, start=1)
                ]
            )
        return hits, retrieval_ms

    def _generate(
        self, messages: list[dict], name: str = "llm"
    ) -> tuple[object, str, int, float]:
        start = time.perf_counter()
        response, provider, attempts = chat_with_failover(
            clients=self._clients,
            providers=self._providers,
            model=self._model,
            messages=messages,
            langfuse=self._langfuse,
            name=name,
            sleep=self._sleep,
            max_tokens=self._max_tokens,
        )
        return response, provider, attempts, 1000 * (time.perf_counter() - start)
