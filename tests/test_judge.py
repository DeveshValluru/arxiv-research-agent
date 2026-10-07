from types import SimpleNamespace

import pytest

from arxiv_agent.evals.judge import (
    FALSE_PREMISE_NOTE,
    JUDGE_MODEL,
    JUDGE_PROMPT,
    Judge,
    JudgeError,
    build_judge_messages,
    parse_verdict,
)

GOOD_JSON = '{"reasoning": "Names the same model.", "verdict": "correct"}'


class FakeHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class FakeClient:
    def __init__(self, content: str = GOOD_JSON, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    def chat_completion(self, messages, model=None, max_tokens=None, temperature=None):
        self.calls.append(
            {"messages": messages, "model": model, "temperature": temperature}
        )
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=30),
        )


def test_messages_hold_rubric_references_and_candidate():
    system, user = build_judge_messages(
        "Which model?", ["LLaMA-7B.", "LLaMA 7B"], "It is LLaMA-7B [S1]."
    )
    assert system == {"role": "system", "content": JUDGE_PROMPT}
    assert user["content"] == (
        "Question: Which model?\n\n"
        "Reference answers:\n- LLaMA-7B.\n- LLaMA 7B\n\n"
        "Candidate answer:\nIt is LLaMA-7B [S1]."
    )


def test_false_premise_adds_its_rule():
    system, _ = build_judge_messages("Why worse?", ["It isn't."], "x", True)
    assert system["content"] == JUDGE_PROMPT + FALSE_PREMISE_NOTE


def test_parse_plain_json():
    verdict = parse_verdict(GOOD_JSON)
    assert (verdict.verdict, verdict.reasoning) == ("correct", "Names the same model.")


def test_parse_json_wrapped_in_a_code_fence_and_chatter():
    raw = f"Here is my grade:\n```json\n{GOOD_JSON}\n```\nHope that helps."
    assert parse_verdict(raw).verdict == "correct"


def test_unknown_verdict_is_rejected():
    with pytest.raises(JudgeError, match="unusable"):
        parse_verdict('{"reasoning": "close enough", "verdict": "mostly_right"}')


def test_missing_json_is_rejected():
    with pytest.raises(JudgeError, match="no JSON"):
        parse_verdict("The answer is correct.")


def make_judge(*clients: FakeClient, sleeps: list | None = None) -> Judge:
    providers = [f"fake-{letter}" for letter in "ab"[: len(clients)]]
    return Judge(
        providers=providers,
        clients=dict(zip(providers, clients)),
        sleep=(sleeps if sleeps is not None else []).append,
    )


def test_grade_calls_the_judge_model_deterministically():
    client = FakeClient()

    verdict = make_judge(client).grade("Which model?", ["LLaMA-7B."], "LLaMA-7B")

    assert verdict.verdict == "correct"
    assert client.calls[0]["model"] == JUDGE_MODEL
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["messages"] == build_judge_messages(
        "Which model?", ["LLaMA-7B."], "LLaMA-7B"
    )


def test_busy_judge_provider_fails_over_to_the_next():
    busy, backup = FakeClient(error=FakeHTTPError(429)), FakeClient()

    verdict = make_judge(busy, backup).grade("Which model?", ["LLaMA-7B."], "LLaMA-7B")

    assert verdict.verdict == "correct"
    assert (len(busy.calls), len(backup.calls)) == (2, 1)


def test_unreachable_judge_becomes_a_judge_error():
    judge = make_judge(FakeClient(error=FakeHTTPError(503)))

    with pytest.raises(JudgeError, match="judge call failed"):
        judge.grade("Which model?", ["LLaMA-7B."], "LLaMA-7B")
