import os

import feedparser
import httpx

ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _build_user_agent() -> str:
    name = os.environ.get("ARXIV_CONTACT_NAME", "")
    email = os.environ.get("ARXIV_CONTACT_EMAIL", "")
    contact = [part for part in (name, f"mailto:{email}" if email else "") if part]
    if not contact:
        return "arxiv-agent/0.1"
    return f"arxiv-agent/0.1 ({'; '.join(contact)})"


USER_AGENT = _build_user_agent()


def search_papers(query: str, max_results: int = 5) -> list[dict]:
    response = httpx.get(
        ARXIV_API_URL,
        params={"search_query": query, "max_results": max_results},
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(10.0, read=60.0),
    )
    response.raise_for_status()
    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        papers.append({"id": entry.id, "title": entry.title})
    return papers


if __name__ == "__main__":
    for paper in search_papers('abs:"LLM as a judge"'):
        print(paper["id"], "|", paper["title"])
