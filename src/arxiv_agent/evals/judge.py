import re
from typing import Literal

from huggingface_hub import InferenceClient
from langfuse import Langfuse, get_client
from pydantic import BaseModel, ValidationError

# The judge must not share a family with the generator (Qwen): models favour
# text that sounds like their own (self-enhancement bias).
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
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
        provider: str = "novita",
        client: InferenceClient | None = None,
        langfuse: Langfuse | None = None,
    ) -> None:
        self._model = model
        self._client = client or InferenceClient(provider=provider)
        self._langfuse = langfuse or get_client()

    def grade(
        self,
        question: str,
        gold_answers: list[str],
        answer: str,
        false_premise: bool = False,
    ) -> Verdict:
        messages = build_judge_messages(question, gold_answers, answer, false_premise)
        with self._langfuse.start_as_current_observation(
            as_type="generation",
            name="judge",
            model=self._model,
            input=messages,
            metadata={"judge_prompt_version": JUDGE_PROMPT_VERSION},
        ) as generation:
            # temperature 0: the same answer should get the same grade every run
            response = self._client.chat_completion(
                messages, model=self._model, max_tokens=300, temperature=0.0
            )
            raw = response.choices[0].message.content or ""
            generation.update(
                output=raw,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
            )
        return parse_verdict(raw)
