import logging
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
MIN_REQUEST_INTERVAL_SECONDS = 3.0
METADATA_BATCH_SIZE = 50
ARXIV_ID_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$"
)

logger = logging.getLogger(__name__)


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


def _error_message(response: httpx.Response) -> str:
    for entry in feedparser.parse(response.text).entries:
        if entry.get("title") == "Error":
            return entry.get("summary", "")
    return ""


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = http or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(10.0, read=60.0),
        )
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def search_papers(self, query: str, max_results: int = 5) -> list[PaperSummary]:
        xml = self._request({"search_query": query, "max_results": max_results})
        return _parse_feed(xml)

    def get_metadata(self, ids: list[str]) -> dict[str, PaperSummary]:
        valid_ids = [i for i in dict.fromkeys(ids) if ARXIV_ID_PATTERN.match(i)]
        found: dict[str, PaperSummary] = {}
        for start in range(0, len(valid_ids), METADATA_BATCH_SIZE):
            batch = valid_ids[start : start + METADATA_BATCH_SIZE]
            xml = self._request({"id_list": ",".join(batch), "max_results": len(batch)})
            for paper in _parse_feed(xml):
                found[paper.arxiv_id] = paper
        return found

    def _pace(self) -> None:
        if self._last_request_at is not None:
            elapsed = self._clock() - self._last_request_at
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                self._sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_at = self._clock()

    def _request(self, params: dict[str, str | int]) -> str:
        problem = ""
        for attempt in range(1, self._max_attempts + 1):
            wait = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            self._pace()
            try:
                response = self._http.get(ARXIV_API_URL, params=params)
            except httpx.TransportError as exc:
                problem = f"network error ({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    return response.text
                if response.status_code not in RETRYABLE_STATUSES:
                    message = f"arXiv rejected the request: HTTP {response.status_code}"
                    detail = _error_message(response)
                    if detail:
                        message += f": {detail}"
                    raise ArxivQueryError(message)
                problem = f"HTTP {response.status_code}"
                wait = _retry_after_seconds(response) or wait
            if attempt < self._max_attempts:
                logger.warning(
                    "arXiv %s on attempt %d/%d; retrying in %.0f s",
                    problem,
                    attempt,
                    self._max_attempts,
                    wait,
                )
                self._sleep(wait)
        raise ArxivUnavailableError(
            f"arXiv unavailable after {self._max_attempts} attempts: {problem}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = ArxivClient()
    try:
        for paper in client.search_papers('abs:"LLM as a judge"'):
            print(paper.arxiv_id, f"v{paper.version}", "|", paper.title)
    except ArxivUnavailableError as exc:
        print(f"arXiv is unavailable right now: {exc}")
