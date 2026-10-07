import asyncio
import json

import pytest

from arxiv_agent.mcp_servers.arxiv_server import create_server as arxiv_server
from arxiv_agent.tools.toolbox import McpToolbox, ToolResult
from tests.mcp_fakes import FakeArxiv, make_toolbox


def with_toolbox(use):
    async def go():
        async with make_toolbox() as toolbox:
            return await use(toolbox)

    return asyncio.run(go())


def test_one_flat_list_of_every_servers_tools():
    async def use(toolbox):
        return toolbox.tool_names, toolbox.tool_specs()

    names, specs = with_toolbox(use)

    assert names == [
        "search_papers",
        "get_metadata",
        "get_references",
        "get_paper",
        "get_citations",
    ]
    assert all(spec["type"] == "function" for spec in specs)
    paper = next(s["function"] for s in specs if s["function"]["name"] == "get_paper")
    assert "citation count" in paper["description"]
    assert paper["parameters"]["required"] == ["arxiv_id"]


def test_calls_go_to_the_server_that_owns_the_tool():
    async def use(toolbox):
        return (
            await toolbox.call("get_paper", {"arxiv_id": "2306.05685"}),
            await toolbox.call("get_metadata", {"arxiv_ids": ["2306.05685"]}),
        )

    paper, metadata = with_toolbox(use)

    assert (paper.server, paper.ok) == ("openalex", True)
    assert paper.arguments == {"arxiv_id": "2306.05685"}
    assert paper.content["cited_by_count"] == 487
    assert (metadata.server, metadata.ok) == ("arxiv", True)
    assert metadata.content["not_found"] == ["2306.05685"]


def test_a_tool_error_comes_back_as_a_result():
    async def use(toolbox):
        return await toolbox.call("get_paper", {"arxiv_id": "2411.99999"})

    result = with_toolbox(use)

    assert not result.ok
    assert "no record of arXiv paper 2411.99999" in result.content
    assert result.for_model().startswith("Error: ")


def test_an_unknown_tool_lists_the_real_ones():
    async def use(toolbox):
        return await toolbox.call("delete_everything", {})

    result = with_toolbox(use)

    assert (result.ok, result.server) == (False, None)
    assert "available: ['search_papers'" in result.content


def test_two_servers_offering_the_same_tool_is_refused():
    async def go():
        async with McpToolbox(
            {"one": arxiv_server(FakeArxiv()), "two": arxiv_server(FakeArxiv())}
        ):
            pass

    with pytest.raises(ValueError, match="offered by both 'one' and 'two'"):
        asyncio.run(go())


def test_results_reach_the_model_as_json():
    result = ToolResult(
        tool="get_paper", server="openalex", ok=True, content={"a": 1}, duration_ms=1.0
    )
    assert json.loads(result.for_model()) == {"a": 1}
