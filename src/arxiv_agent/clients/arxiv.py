import os
import re
import time
from collections.abc import Callable
from datetime import datetime

import feedparser
import httpx
from pydantic import BaseModel, ConfigDict

ARXIV_API_URL = "https://export.arxiv.org/api/query"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
BACKOFF_BASE_SECONDS = 5.0


def _build_user_agent() -> str:
    name = os.environ.get("ARXIV_CONTACT_NAME", "")
    email = os.environ.get("ARXIV_CONTACT_EMAIL", "")
    contact = [part for part in (name, f"mailto:{email}" if email else "") if part]
    if not contact:
        return "arxiv-agent/0.1"
    return f"arxiv-agent/0.1 ({'; '.join(contact)})"


USER_AGENT = _build_user_agent()


class ArxivError(Exception):
    pass


class ArxivQueryError(ArxivError):
    pass


class ArxivUnavailableError(ArxivError):
    pass


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


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None or not value.isdigit():
        return None
    return float(value)


class ArxivClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(10.0, read=60.0),
        )
        self._max_attempts = max_attempts
        self._sleep = sleep

    def search_papers(self, query: str, max_results: int = 5) -> list[PaperSummary]:
        xml = self._request({"search_query": query, "max_results": max_results})
        return _parse_feed(xml)

    def _request(self, params: dict[str, str | int]) -> str:
        problem = ""
        for attempt in range(1, self._max_attempts + 1):
            wait = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            try:
                response = self._http.get(ARXIV_API_URL, params=params)
            except httpx.TransportError as exc:
                problem = f"network error ({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    return response.text
                if response.status_code not in RETRYABLE_STATUSES:
                    raise ArxivQueryError(
                        f"arXiv rejected the request: HTTP {response.status_code}"
                    )
                problem = f"HTTP {response.status_code}"
                wait = _retry_after_seconds(response) or wait
            if attempt < self._max_attempts:
                self._sleep(wait)
        raise ArxivUnavailableError(
            f"arXiv unavailable after {self._max_attempts} attempts: {problem}"
        )


if __name__ == "__main__":
    client = ArxivClient()
    for paper in client.search_papers('abs:"LLM as a judge"'):
        print(paper.arxiv_id, f"v{paper.version}", "|", paper.title)
