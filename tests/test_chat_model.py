import asyncio
from types import SimpleNamespace

from arxiv_agent.llm import ChatModel, Usage


class FakeClient:
    def __init__(self, cost: float | None) -> None:
        self.cost = cost

    def chat_completion(self, messages, model, **params):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<think>hmm</think>Hello.")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=12, completion_tokens=3, estimated_cost=self.cost
            ),
        )


def complete(cost: float | None):
    model = ChatModel("some/model", ["p"], clients={"p": FakeClient(cost)})
    return asyncio.run(model.complete([{"role": "user", "content": "Hi"}], name="t"))


def test_complete_returns_the_text_and_what_the_call_used():
    completion = complete(cost=0.0002)

    assert completion.text == "Hello."  # thinking stripped
    assert completion.usage == Usage(
        llm_calls=1, prompt_tokens=12, completion_tokens=3, cost_usd=0.0002
    )


def test_a_provider_that_reports_no_cost_adds_zero():
    assert complete(cost=None).usage.cost_usd == 0.0
