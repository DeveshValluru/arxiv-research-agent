"""Fake arXiv and OpenAlex clients behind the real MCP servers, for tests.

The servers, the MCP protocol and the toolbox are all real; only the network
calls to arXiv and OpenAlex are replaced.
"""

from arxiv_agent.clients.openalex import Work
from arxiv_agent.mcp_servers.arxiv_server import create_server as arxiv_server
from arxiv_agent.mcp_servers.openalex_server import create_server as openalex_server
from arxiv_agent.tools.toolbox import McpToolbox

MT_BENCH = Work(
    openalex_id="W4380353763",
    title="Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
    year=2023,
    cited_by_count=487,
    venue="arXiv (Cornell University)",
    doi="10.48550/arxiv.2306.05685",
    arxiv_id="2306.05685",
    authors=["Lianmin Zheng"],
    author_count=13,
)


class FakeArxiv:
    def search_papers(self, query, max_results=5):
        return []

    def get_metadata(self, ids):
        return {}

    def fetch_html(self, arxiv_id, version=None):
        return None


class FakeOpenAlex:
    def get_work_by_arxiv_id(self, arxiv_id):
        return MT_BENCH if arxiv_id == "2306.05685" else None

    def get_citations(self, openalex_id, limit=10, sort="most_cited"):
        return 486, []


def make_toolbox() -> McpToolbox:
    return McpToolbox(
        {
            "arxiv": arxiv_server(FakeArxiv()),
            "openalex": openalex_server(FakeOpenAlex()),
        }
    )
