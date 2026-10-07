import time
from collections.abc import Callable

from huggingface_hub import InferenceClient
from langfuse import Langfuse, get_client, propagate_attributes
from pydantic import BaseModel, ConfigDict

from arxiv_agent.guardrails.output import OutputGuard
from arxiv_agent.llm import chat_with_failover
from arxiv_agent.qa.checker import REFUSAL, CheckedAnswer, check_answer
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

NO_TRACE_ID = "0" * 32  # what a disabled Langfuse client reports


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    chunk_id: str
    section_path: list[str]
    score: float
    text: str


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
    ) -> None:
        if not providers:
            raise ValueError("need at least one provider")
        self._retriever = retriever
        self._model = model
        self._providers = providers
        self._k = k
        self._max_tokens = max_tokens
        self._clients = clients or {p: InferenceClient(provider=p) for p in providers}
        self._langfuse = langfuse or get_client()
        self._sleep = sleep
        self._guard = OutputGuard([SYSTEM_PROMPT])

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
            prompt_version=PROMPT_VERSION,
            retriever=self._retriever.name,
            chunker_version=hits[0].chunk.chunker_version,
            trace_id=None if trace_id == NO_TRACE_ID else trace_id,
        )

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
                raise LookupError(
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

    def _generate(self, messages: list[dict]) -> tuple[object, str, int, float]:
        start = time.perf_counter()
        response, provider, attempts = chat_with_failover(
            clients=self._clients,
            providers=self._providers,
            model=self._model,
            messages=messages,
            langfuse=self._langfuse,
            sleep=self._sleep,
            max_tokens=self._max_tokens,
        )
        return response, provider, attempts, 1000 * (time.perf_counter() - start)
