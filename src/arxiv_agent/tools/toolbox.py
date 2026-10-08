"""Connect to MCP servers as a client and call their tools, traced in Langfuse.

One McpToolbox holds a connection to each server and routes every call to the
server that owns the tool, so an agent sees one flat list of tools.
"""

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any, Self

import anyio
from langfuse import Langfuse, get_client
from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field

SERVERS = {
    "arxiv": "arxiv_agent.mcp_servers.arxiv_server",
    "openalex": "arxiv_agent.mcp_servers.openalex_server",
}
# Stdio servers get a minimal environment; pass only the contact details
# (arXiv's User-Agent, OpenAlex's polite pool), never other secrets.
PASS_THROUGH = ("ARXIV_CONTACT_NAME", "ARXIV_CONTACT_EMAIL")
# The one exception: the arXiv server takes turns with every other arXiv
# client through the database (clients/pacing.py), so it gets DATABASE_URL.
SERVER_ENV = {"arxiv": ("DATABASE_URL",)}
# What a tool call can fail with besides a tool error: a protocol error, a
# server that died (closed or broken stream), a timeout, or a broken pipe.
CALL_FAILURES = (
    MCPError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    TimeoutError,
    OSError,
)


def stdio_server(module: str, extra: tuple[str, ...] = ()) -> StdioServerParameters:
    names = (*PASS_THROUGH, *extra)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=get_default_environment()
        | {name: os.environ[name] for name in names if name in os.environ},
    )


class ToolResult(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)
    server: str | None
    ok: bool
    content: dict | str  # structured output, or the error text
    duration_ms: float

    def for_model(self) -> str:
        # What goes back to the LLM: JSON for results, plain text for errors.
        if isinstance(self.content, dict):
            return json.dumps(self.content, ensure_ascii=False)
        return self.content if self.ok else f"Error: {self.content}"


class McpToolbox:
    # servers maps a name to anything mcp.Client accepts: stdio parameters
    # (a subprocess), a URL (Streamable HTTP), or an in-process server (tests).
    def __init__(
        self, servers: dict[str, Any] | None = None, langfuse: Langfuse | None = None
    ) -> None:
        self._servers = servers or {
            name: stdio_server(module, SERVER_ENV.get(name, ()))
            for name, module in SERVERS.items()
        }
        self._langfuse = langfuse or get_client()
        self._stack = AsyncExitStack()
        self._clients: dict[str, Client] = {}
        self._owner: dict[str, str] = {}  # tool name -> server name
        self._tools: list = []

    async def __aenter__(self) -> Self:
        for name, target in self._servers.items():
            client = await self._stack.enter_async_context(Client(target))
            self._clients[name] = client
            for tool in (await client.list_tools()).tools:
                if tool.name in self._owner:
                    raise ValueError(
                        f"tool {tool.name!r} is offered by both "
                        f"{self._owner[tool.name]!r} and {name!r}"
                    )
                self._owner[tool.name] = name
                self._tools.append(tool)
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.aclose()

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    def tool_specs(self) -> list[dict]:
        # The function-calling format chat-completion APIs accept: the model
        # sees exactly the name, description and schema each server declared.
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools
        ]

    async def call(self, name: str, arguments: dict) -> ToolResult:
        server = self._owner.get(name)
        with self._langfuse.start_as_current_observation(
            as_type="tool", name=name, input=arguments, metadata={"server": server}
        ) as span:
            start = time.perf_counter()
            ok, content = await self._run(server, name, arguments)
            result = ToolResult(
                tool=name,
                arguments=arguments,
                server=server,
                ok=ok,
                content=content,
                duration_ms=1000 * (time.perf_counter() - start),
            )
            span.update(
                output=content,
                level=None if ok else "ERROR",
                status_message=None if ok else str(content)[:500],
            )
        return result

    async def _run(
        self, server: str | None, name: str, arguments: dict
    ) -> tuple[bool, dict | str]:
        # Every failure becomes a result the model can read and recover from,
        # never an exception that ends the agent's run.
        if server is None:
            return False, f"unknown tool {name!r}; available: {self.tool_names}"
        try:
            response = await self._clients[server].call_tool(name, arguments)
        except CALL_FAILURES as exc:
            return False, f"tool call failed: {type(exc).__name__}: {exc}"
        if response.is_error or response.structured_content is None:
            text = "\n".join(c.text for c in response.content if hasattr(c, "text"))
            return not response.is_error, text
        return True, response.structured_content
