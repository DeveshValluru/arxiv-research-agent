"""arXiv's HTML pages, cached on disk.

Fetching a page is the slow, rate-limited step, so everything that reads arXiv
HTML (ingestion, and the arXiv MCP server's bibliographies) goes through this
cache: a page fetched once is never fetched again.
"""

from pathlib import Path

from arxiv_agent.clients.arxiv import ArxivClient

HTML_CACHE_DIR = Path("data/html")


def load_html(
    client: ArxivClient, arxiv_id: str, version: int, cache_dir: Path = HTML_CACHE_DIR
) -> str | None:
    # Keyed by version: a bare id means "latest", which changes over time.
    path = cache_dir / f"{arxiv_id.replace('/', '_')}v{version}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")

    html = client.fetch_html(arxiv_id, version)
    if html is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html
