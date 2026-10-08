"""Claim support for Q&A answers: does each sentence say what its sources say?

The review's Critic checks every sentence (6.3); answers had only format
checks. Same checks here, measured in 6.3: result numbers in code first, then
the judge (a different model family from the answerer) with the Critic's
prompt, one call per sentence, in parallel.

A sentence that fails is removed; if nothing supported is left, the answer
refuses. Measured on the Q&A eval (6.3b): 64 cited sentences judged, 2 removed.
One removal was right but cost a correct answer: the sentence was true apart
from one detail its source didn't contain, and removing it left nothing. So the
Answerer first asks for one rewrite of the failing sentences (6.3c, in
qa/answerer.py) and removes only what still fails.
"""

import contextvars
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from huggingface_hub import InferenceClient
from langfuse import Langfuse, get_client
from pydantic import BaseModel, ConfigDict

from arxiv_agent.evals.judge import JUDGE_MODEL
from arxiv_agent.llm import (
    JUDGE_TIMEOUT,
    LLMOutputError,
    LLMUnavailableError,
    chat_with_failover,
    parse_json_object,
)
from arxiv_agent.qa.checker import CITATION, REFUSAL, CheckedAnswer
from arxiv_agent.review.citations import split_sentences
from arxiv_agent.review.critic import CRITIC_PROMPT, judging_request
from arxiv_agent.review.numbers import missing_numbers
from arxiv_agent.review.state import Claim, Judgment
from arxiv_agent.storage.chunk_store import SearchHit

# The review's judge order: fastest first, measured with parallel calls (5.1b).
SUPPORT_PROVIDERS = ["together", "novita", "ovhcloud"]
FAILING = {"overstated", "unsupported"}


class SentenceSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence: str
    sources: list[int]  # the [S#] it cites
    # uncited: states no sourced fact (e.g. "the paper doesn't cover X");
    # unchecked: the judge failed, which doesn't make the sentence wrong
    verdict: Literal["supported", "overstated", "unsupported", "unchecked", "uncited"]
    reason: str


class SupportChecker:
    def __init__(
        self,
        model: str = JUDGE_MODEL,
        providers: list[str] = SUPPORT_PROVIDERS,
        clients: dict[str, InferenceClient] | None = None,
        langfuse: Langfuse | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_workers: int = 4,
    ) -> None:
        self._model = model
        self._providers = providers
        # Judge calls are short (~2 s): a provider taking 30 s counts as down.
        self._clients = clients or {
            p: InferenceClient(provider=p, timeout=JUDGE_TIMEOUT) for p in providers
        }
        self._langfuse = langfuse or get_client()
        self._sleep = sleep
        self._max_workers = max_workers

    def check(
        self,
        text: str,
        hits: list[SearchHit],
        known: Iterable[SentenceSupport] = (),
    ) -> list[SentenceSupport]:
        # known: verdicts from an earlier check against the same sources. A
        # rewrite keeps most sentences word for word; only new ones are judged.
        seen = {s.sentence: s for s in known}
        sentences = [
            s for paragraph in text.split("\n") for s in split_sentences(paragraph)
        ]
        # Each worker thread runs in a copy of this context, so its judge
        # calls nest under the current trace span.
        with ThreadPoolExecutor(self._max_workers) as pool:
            futures = [
                None
                if s in seen
                else pool.submit(
                    contextvars.copy_context().run, self._check_one, s, hits
                )
                for s in sentences
            ]
            return [
                seen[s] if future is None else future.result()
                for s, future in zip(sentences, futures, strict=True)
            ]

    def _check_one(self, sentence: str, hits: list[SearchHit]) -> SentenceSupport:
        sources = list(dict.fromkeys(int(n) for n in CITATION.findall(sentence)))
        cited = [
            Claim(
                label=f"S{n}",
                arxiv_id=hits[n - 1].chunk.arxiv_id,
                version=hits[n - 1].chunk.version,
                chunk_id=hits[n - 1].chunk.chunk_id,
                section=" > ".join(hits[n - 1].chunk.section_path),
                claim="",
                quote="",
                passage=hits[n - 1].chunk.text,
            )
            for n in sources
            if 1 <= n <= len(hits)  # check_answer already flags the others
        ]

        def support(verdict: str, reason: str) -> SentenceSupport:
            return SentenceSupport(
                sentence=sentence, sources=sources, verdict=verdict, reason=reason
            )

        if not cited:
            return support("uncited", "cites no source")
        missing = missing_numbers(
            CITATION.sub("", sentence), [claim.passage for claim in cited]
        )
        if missing:
            return support(
                "unsupported", f"{', '.join(missing)} isn't in the sources it cites"
            )
        try:
            response, _, _ = chat_with_failover(
                clients=self._clients,
                providers=self._providers,
                model=self._model,
                messages=[
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": judging_request(sentence, cited)},
                ],
                langfuse=self._langfuse,
                name="support-judge",
                sleep=self._sleep,
                max_tokens=150,
            )
            judgment = parse_json_object(
                response.choices[0].message.content or "", Judgment
            )
        except (LLMOutputError, LLMUnavailableError) as exc:
            return support("unchecked", f"judge failed: {exc}"[:200])
        return support(judgment.verdict, judgment.reason)


def apply_support(
    answer: CheckedAnswer, support: list[SentenceSupport]
) -> CheckedAnswer:
    # Keep the sentences that passed; if no cited sentence is left, refuse.
    failed = [s for s in support if s.verdict in FAILING]
    if not failed:
        return answer
    kept = [s for s in support if s.verdict not in FAILING]
    problems = [
        *answer.problems,
        *(f"removed ({s.verdict}: {s.reason}): {s.sentence}" for s in failed),
    ]
    if not any(s.verdict != "uncited" for s in kept):
        return CheckedAnswer(
            text=REFUSAL, status="refused", cited=[], problems=problems
        )
    text = " ".join(s.sentence for s in kept)
    cited = list(dict.fromkeys(n for s in kept for n in s.sources))
    return CheckedAnswer(
        text=text, status=answer.status, cited=cited, problems=problems
    )
