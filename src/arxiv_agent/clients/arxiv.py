import os
import re
from datetime import datetime

import feedparser
import httpx
from pydantic import BaseModel, ConfigDict

ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _build_user_agent() -> str:
    name = os.environ.get("ARXIV_CONTACT_NAME", "")
    email = os.environ.get("ARXIV_CONTACT_EMAIL", "")
    contact = [part for part in (name, f"mailto:{email}" if email else "") if part]
    if not contact:
        return "arxiv-agent/0.1"
    return f"arxiv-agent/0.1 ({'; '.join(contact)})"


USER_AGENT = _build_user_agent()


class PaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    primary_category: str
    categories: list[str]


def _split_id(raw_id: str) -> tuple[str, int]:
    match = re.search(r"abs/(.+)v(\d+)$", raw_id)
    if match is None:
        raise ValueError(f"unexpected arXiv id: {raw_id}")
    return match.group(1), int(match.group(2))


def _clean(text: str) -> str:
    return " ".join(text.split())


def _parse_feed(xml_text: str) -> list[PaperSummary]:
    feed = feedparser.parse(xml_text)
    papers = []

    for entry in feed.entries:
        arxiv_id, version = _split_id(entry.id)
        papers.append(
            PaperSummary(
                arxiv_id=arxiv_id,
                version=version,
                title=_clean(entry.title),
                authors=[author["name"] for author in entry.authors],
                abstract=_clean(entry.summary),
                published=entry.published,
                updated=entry.updated,
                primary_category=entry.arxiv_primary_category["term"],
                categories=[tag["term"] for tag in entry.tags],
            )
        )
    return papers


def search_papers(query: str, max_results: int = 5) -> list[PaperSummary]:
    response = httpx.get(
        ARXIV_API_URL,
        params={"search_query": query, "max_results": max_results},
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(10.0, read=60.0),
    )
    response.raise_for_status()
    return _parse_feed(response.text)


if __name__ == "__main__":
    for paper in search_papers('abs:"LLM as a judge"'):
        print(paper.arxiv_id, f"v{paper.version}", "|", paper.title)
