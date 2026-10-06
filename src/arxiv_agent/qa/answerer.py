import time

from huggingface_hub import InferenceClient
from pydantic import BaseModel, ConfigDict

from arxiv_agent.ingestion.embedder import Embedder, HostedEmbedder
from arxiv_agent.qa.checker import REFUSAL, CheckedAnswer, check_answer
from arxiv_agent.storage.chunk_store import ChunkStore, SearchHit

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
    provider: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    retrieval_ms: float
    generation_ms: float
    prompt_version: int
    embedder: str
    chunker_version: int


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
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder | HostedEmbedder,
        model: str,
        provider: str,
        k: int = 5,
        max_tokens: int = 500,
        client: InferenceClient | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._model = model
        self._provider = provider
        self._k = k
        self._max_tokens = max_tokens
        self._client = client or InferenceClient(provider=provider)

    def ask(self, question: str, arxiv_id: str, version: int) -> QAResult:
        start = time.perf_counter()
        hits = self._store.vector_search(
            self._embedder.model_id,
            self._embedder.embed_query(question),
            k=self._k,
            papers=[(arxiv_id, version)],
        )
        retrieval_ms = 1000 * (time.perf_counter() - start)
        if not hits:
            raise LookupError(
                f"{arxiv_id}v{version} has no indexed chunks for "
                f"{self._embedder.model_id}; ingest it first"
            )

        start = time.perf_counter()
        response = self._client.chat_completion(
            build_messages(question, hits),
            model=self._model,
            max_tokens=self._max_tokens,
        )
        generation_ms = 1000 * (time.perf_counter() - start)

        choice = response.choices[0]
        answer = check_answer(choice.message.content or "", n_sources=len(hits))
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
            provider=self._provider,
            finish_reason=choice.finish_reason,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=getattr(response.usage, "estimated_cost", None),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            prompt_version=PROMPT_VERSION,
            embedder=self._embedder.model_id,
            chunker_version=hits[0].chunk.chunker_version,
        )
