"""A minimal tool-calling loop: the LLM picks tools, code runs them, repeat.

This is the core of every agent. Phase 5 rebuilds it in LangGraph with state,
planning and several agents; here it is in its simplest form, with a cap on
tool calls so a confused model can't loop forever.
"""

import asyncio
import json
from typing import Literal

from huggingface_hub import InferenceClient
from langfuse import Langfuse, get_client
from pydantic import BaseModel

from arxiv_agent.llm import ANSWER_TIMEOUT, chat_with_failover
from arxiv_agent.qa.answerer import NO_TRACE_ID
from arxiv_agent.qa.checker import THINKING
from arxiv_agent.tools.toolbox import McpToolbox, ToolResult

MAX_TOOL_CALLS = 6
SYSTEM_PROMPT = (
    "You are a research assistant with tools for arXiv search and citation data. "
    "Use the tools to answer; never guess arXiv ids, titles or citation counts. "
    "When you have enough information, answer briefly and name the arXiv ids or "
    "DOIs you relied on. /no_think"
)


class LoopResult(BaseModel):
    answer: str
    tool_calls: list[ToolResult]
    llm_calls: int
    stopped_by: Literal["answer", "call_limit"]
    trace_id: str | None


def _parse_arguments(raw: str | dict) -> dict | None:
    # Providers send arguments as a JSON string; models sometimes write bad JSON.
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def run_tool_loop(
    question: str,
    toolbox: McpToolbox,
    model: str,
    providers: list[str],
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_tokens: int = 800,
    clients: dict[str, InferenceClient] | None = None,
    langfuse: Langfuse | None = None,
) -> LoopResult:
    langfuse = langfuse or get_client()
    clients = clients or {
        p: InferenceClient(provider=p, timeout=ANSWER_TIMEOUT) for p in providers
    }
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    results: list[ToolResult] = []
    llm_calls = 0

    with langfuse.start_as_current_observation(
        as_type="agent", name="tool-loop", input={"question": question}
    ) as root:
        while True:
            budget_left = max_tool_calls - len(results)
            # The tools stay listed (the history refers to them), but once the
            # budget is spent tool_choice="none" makes the model answer.
            response, _, _ = await asyncio.to_thread(
                chat_with_failover,
                clients=clients,
                providers=providers,
                model=model,
                messages=messages,
                langfuse=langfuse,
                tools=toolbox.tool_specs(),
                tool_choice="auto" if budget_left > 0 else "none",
                max_tokens=max_tokens,
            )
            llm_calls += 1
            message = response.choices[0].message

            # Stop on an answer, and also when the budget is spent even if the
            # model still asks for tools: the loop must always end.
            if not message.tool_calls or budget_left <= 0:
                answer = THINKING.sub("", message.content or "").strip()
                stopped_by = "answer" if budget_left > 0 else "call_limit"
                root.update(
                    output=answer,
                    metadata={
                        "tool_calls": len(results),
                        "llm_calls": llm_calls,
                        "stopped_by": stopped_by,
                    },
                )
                trace_id = root.trace_id
                return LoopResult(
                    answer=answer,
                    tool_calls=results,
                    llm_calls=llm_calls,
                    stopped_by=stopped_by,
                    trace_id=None if trace_id == NO_TRACE_ID else trace_id,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments
                                if isinstance(call.function.arguments, str)
                                else json.dumps(call.function.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            # Every tool call needs an answering tool message, even skipped ones.
            for call in message.tool_calls:
                if len(results) >= max_tool_calls:
                    content = "Skipped: the tool-call budget is spent. Answer now."
                else:
                    arguments = _parse_arguments(call.function.arguments)
                    if arguments is None:
                        result = ToolResult(
                            tool=call.function.name,
                            arguments={},
                            server=None,
                            ok=False,
                            content="the arguments were not valid JSON; try again",
                            duration_ms=0.0,
                        )
                    else:
                        result = await toolbox.call(call.function.name, arguments)
                    results.append(result)
                    content = result.for_model()
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )
