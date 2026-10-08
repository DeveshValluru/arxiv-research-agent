import re
import time
from collections.abc import Callable
from typing import Literal

from huggingface_hub import InferenceClient
from langfuse import Langfuse, get_client
from pydantic import BaseModel, ValidationError

from arxiv_agent.llm import JUDGE_TIMEOUT, chat_with_failover

# The judge must not share a family with the generator (Qwen): models favour
# text that sounds like their own (self-enhancement bias).
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
JUDGE_PROVIDERS = ["novita", "ovhcloud"]
JUDGE_PROMPT_VERSION = 1
SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_PROMPT = """You grade answers to questions about a research paper.
You get the question, one or more reference answers written by experts, and a candidate answer.
Judge only whether the candidate agrees with the reference answers.

Verdicts:
- correct: contains the key information of at least one reference answer and nothing that contradicts them.
- partially_correct: contains some of the key information but misses an important part.
- incorrect: misses the key information, contradicts the references, or says the paper does not answer.

Rules:
- Do not reward length or extra detail. A short answer with the key information is correct.
- Extra details that are not in the references are fine unless they contradict them.
- Ignore citation marks like [S2] and differences in wording.

Reply with only a JSON object, reasoning first:
{"reasoning": "<one sentence>", "verdict": "correct" | "partially_correct" | "incorrect"}"""

FALSE_PREMISE_NOTE = """

This question contains a false premise; the reference answer corrects it.
- correct: the candidate rejects or corrects the false premise.
- incorrect: the candidate accepts the premise as true."""


class Verdict(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reasoning: str


class JudgeError(Exception):
    pass


def build_judge_messages(
    question: str, gold_answers: list[str], answer: str, false_premise: bool = False
) -> list[dict]:
    references = "\n".join(f"- {gold}" for gold in gold_answers)
    return [
        {
            "role": "system",
            "content": JUDGE_PROMPT + (FALSE_PREMISE_NOTE if false_premise else ""),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Reference answers:\n{references}\n\n"
                f"Candidate answer:\n{answer}"
            ),
        },
    ]


def parse_verdict(raw: str) -> Verdict:
    # Models wrap JSON in code fences or add a sentence around it, so take the
    # outermost {...}. Be lenient about the wrapping, strict about the verdict.
    match = JSON_OBJECT.search(raw)
    if match is None:
        raise JudgeError(f"no JSON object in judge output: {raw[:200]!r}")
    try:
        return Verdict.model_validate_json(match.group())
    except ValidationError as exc:
        raise JudgeError(f"unusable judge output: {raw[:200]!r}") from exc


class Judge:
    def __init__(
        self,
        model: str = JUDGE_MODEL,
        providers: list[str] | None = None,
        clients: dict[str, InferenceClient] | None = None,
        langfuse: Langfuse | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self._providers = providers or JUDGE_PROVIDERS
        self._clients = clients or {
            provider: InferenceClient(provider=provider, timeout=JUDGE_TIMEOUT)
            for provider in self._providers
        }
        self._langfuse = langfuse or get_client()
        self._sleep = sleep

    def grade(
        self,
        question: str,
        gold_answers: list[str],
        answer: str,
        false_premise: bool = False,
    ) -> Verdict:
        messages = build_judge_messages(question, gold_answers, answer, false_premise)
        try:
            response, _, _ = chat_with_failover(
                clients=self._clients,
                providers=self._providers,
                model=self._model,
                messages=messages,
                langfuse=self._langfuse,
                name="judge",
                metadata={"judge_prompt_version": JUDGE_PROMPT_VERSION},
                sleep=self._sleep,
                max_tokens=300,
                temperature=0.0,  # the same answer should get the same grade
            )
        except Exception as exc:
            # One unreachable judge must not end a 50-question run: the runner
            # counts it as a judge failure and moves on.
            raise JudgeError(f"judge call failed: {exc}") from exc
        return parse_verdict(response.choices[0].message.content or "")
