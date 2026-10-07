import asyncio
import json
from types import SimpleNamespace

from arxiv_agent.tools.tool_loop import run_tool_loop
from tests.mcp_fakes import make_toolbox

GET_PAPER = ("get_paper", '{"arxiv_id": "2306.05685"}')


def reply(content: str = "", calls=()) -> SimpleNamespace:
    tool_calls = [
        SimpleNamespace(
            id=f"call_{n}", function=SimpleNamespace(name=name, arguments=arguments)
        )
        for n, (name, arguments) in enumerate(calls)
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or None),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
    )


class FakeLLM:
    # Plays its replies in order; the last one repeats forever.
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat_completion(self, messages, model=None, **params):
        self.calls.append({"messages": [dict(m) for m in messages], **params})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def run(llm: FakeLLM, question: str = "How cited is MT-Bench?", **kwargs):
    async def go():
        async with make_toolbox() as toolbox:
            return await run_tool_loop(
                question,
                toolbox,
                model="test/fake-llm",
                providers=["fake"],
                clients={"fake": llm},
                **kwargs,
            )

    return asyncio.run(go())


def test_the_model_calls_a_tool_reads_the_result_and_answers():
    llm = FakeLLM(reply(calls=[GET_PAPER]), reply("Cited 487 times (2306.05685)."))

    result = run(llm)

    assert result.answer == "Cited 487 times (2306.05685)."
    assert (result.llm_calls, result.stopped_by) == (2, "answer")
    [call] = result.tool_calls
    assert (call.tool, call.server, call.ok) == ("get_paper", "openalex", True)

    first, second = llm.calls
    assert len(first["tools"]) == 5 and first["tool_choice"] == "auto"
    tool_message = second["messages"][-1]
    assert (tool_message["role"], tool_message["tool_call_id"]) == ("tool", "call_0")
    assert json.loads(tool_message["content"])["cited_by_count"] == 487


def test_thinking_is_stripped_from_the_answer():
    result = run(FakeLLM(reply("<think>hmm</think>\n\nDone.")))
    assert result.answer == "Done."
    assert result.tool_calls == []


def test_a_spent_budget_forces_an_answer():
    llm = FakeLLM(
        reply(calls=[GET_PAPER]), reply(calls=[GET_PAPER]), reply("Answer anyway.")
    )

    result = run(llm, max_tool_calls=2)

    assert len(result.tool_calls) == 2
    assert [c["tool_choice"] for c in llm.calls] == ["auto", "auto", "none"]
    assert (result.answer, result.stopped_by) == ("Answer anyway.", "call_limit")


def test_the_loop_ends_even_if_the_model_never_stops_asking():
    llm = FakeLLM(reply("still asking", calls=[GET_PAPER]))  # repeats forever

    result = run(llm, max_tool_calls=3)

    assert len(result.tool_calls) == 3
    assert result.llm_calls == 4  # 3 rounds of tools, then the forced stop
    assert result.stopped_by == "call_limit"


def test_calls_beyond_the_budget_are_skipped_but_still_answered():
    llm = FakeLLM(reply(calls=[GET_PAPER] * 3), reply("Done."))

    result = run(llm, max_tool_calls=2)

    assert len(result.tool_calls) == 2
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_0", "call_1", "call_2"]
    assert tool_messages[2]["content"].startswith("Skipped")


def test_bad_json_arguments_are_reported_to_the_model():
    llm = FakeLLM(reply(calls=[("get_paper", "{not json")]), reply("Sorry."))

    result = run(llm)

    assert not result.tool_calls[0].ok
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["content"] == (
        "Error: the arguments were not valid JSON; try again"
    )


def test_tracing_is_off_in_tests():
    assert run(FakeLLM(reply("ok"))).trace_id is None
