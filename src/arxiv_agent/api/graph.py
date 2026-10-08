"""A review's citation graph, from what the snowball step recorded.

Snowballing follows the kept papers' references and citations, and notes on
each new paper how it was found: "cites 2406.07791" (it cites a kept paper)
or "cited by 2406.07791" (a kept paper cites it). Those notes are the edges;
no extra lookups. Search results have no links, so a paper only joins the
graph if it was kept or is linked to one that was.
"""

from pydantic import BaseModel, ConfigDict

CITED_BY = "cited by "
CITES = "cites "


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    title: str
    via: str  # search, snowball or reviewer
    kept: bool
    cited: bool  # cited in the finished review
    score: int | None


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # this paper...
    target: str  # ...cites this one


class Graph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[Node]
    edges: list[Edge]


def citation_graph(result: dict) -> Graph:
    kept = {p["arxiv_id"] for p in result.get("kept", [])}
    cited = {p["arxiv_id"] for p in result.get("references", [])}
    papers = {p["arxiv_id"]: p for p in result.get("dropped", [])}
    papers |= {p["arxiv_id"]: p for p in result.get("kept", [])}

    edges: dict[tuple[str, str], Edge] = {}
    for arxiv_id, paper in papers.items():
        for link in paper.get("found_by", []):
            if link.startswith(CITED_BY):
                source, target = link.removeprefix(CITED_BY), arxiv_id
            elif link.startswith(CITES):
                source, target = arxiv_id, link.removeprefix(CITES)
            else:
                continue  # a search query, not a link
            edges[source, target] = Edge(source=source, target=target)

    linked = {end for edge in edges.values() for end in (edge.source, edge.target)}
    nodes = [
        Node(
            arxiv_id=arxiv_id,
            title=papers.get(arxiv_id, {}).get("title", ""),
            via=papers.get(arxiv_id, {}).get("via", "search"),
            kept=arxiv_id in kept,
            cited=arxiv_id in cited,
            score=papers.get(arxiv_id, {}).get("score"),
        )
        for arxiv_id in sorted(kept | linked)
    ]
    return Graph(nodes=nodes, edges=list(edges.values()))
