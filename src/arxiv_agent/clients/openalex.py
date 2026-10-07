"""OpenAlex: an open citation index, used here for "who cites this paper?".

No API key needed. Sending a contact email (mailto) puts requests in OpenAlex's
"polite pool": 10 requests/second and better reliability.
"""

import logging
import os
import re
import time
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

OPENALEX_API = "https://api.openalex.org"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
MIN_REQUEST_INTERVAL_SECONDS = 0.2  # the polite pool allows 10/s; stay well under
BACKOFF_BASE_SECONDS = 1.0
FIELDS = (
    "id,doi,display_name,publication_year,cited_by_count,"
    "primary_location,locations,authorships"
)
AUTHORS_SHOWN = 3
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_ABS = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")
SORTS = {"most_cited": "cited_by_count:desc", "newest": "publication_date:desc"}

logger = logging.getLogger(__name__)


class OpenAlexError(Exception):
    pass


class OpenAlexUnavailableError(OpenAlexError):
    pass


class Work(BaseModel):
    model_config = ConfigDict(extra="forbid")

    openalex_id: str  # e.g. "W4380353763"
    title: str
    year: int | None
    cited_by_count: int
    venue: str | None
    doi: str | None  # an identifier like "10.1038/...", not a URL
    arxiv_id: str | None
    authors: list[str]  # the first few only
    author_count: int


def _arxiv_id(record: dict) -> str | None:
    # A work is on arXiv if its DOI is an arXiv DOI or one of its copies lives
    # at arxiv.org/abs/<id>. Journal versions often have neither.
    match = ARXIV_DOI.search(record.get("doi") or "")
    if match:
        return match.group(1)
    for location in record.get("locations") or []:
        match = ARXIV_ABS.search(location.get("landing_page_url") or "")
        if match:
            return match.group(1)
    return None


def to_work(record: dict) -> Work:
    source = (record.get("primary_location") or {}).get("source") or {}
    authors = [a["author"]["display_name"] for a in record.get("authorships") or []]
    doi = record.get("doi")
    return Work(
        openalex_id=record["id"].rsplit("/", 1)[-1],
        title=record.get("display_name") or "",
        year=record.get("publication_year"),
        cited_by_count=record.get("cited_by_count") or 0,
        venue=source.get("display_name"),
        doi=doi.removeprefix("https://doi.org/") if doi else None,
        arxiv_id=_arxiv_id(record),
        authors=authors[:AUTHORS_SHOWN],
        author_count=len(authors),
    )


class OpenAlexClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        mailto: str | None = None,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = http or httpx.Client(timeout=httpx.Timeout(10.0, read=30.0))
        self._mailto = (
            mailto if mailto is not None else os.environ.get("ARXIV_CONTACT_EMAIL", "")
        )
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def get_work_by_arxiv_id(self, arxiv_id: str) -> Work | None:
        # arXiv registers a DOI for every paper: 10.48550/arXiv.<id>
        record = self._request(
            f"/works/doi:10.48550/arXiv.{arxiv_id}", {"select": FIELDS}
        )
        return to_work(record) if record else None

    def get_citations(
        self,
        openalex_id: str,
        limit: int = 10,
        sort: Literal["most_cited", "newest"] = "most_cited",
    ) -> tuple[int, list[Work]]:
        page = self._request(
            "/works",
            {
                "filter": f"cites:{openalex_id}",
                "sort": SORTS[sort],
                "per-page": limit,
                "select": FIELDS,
            },
        )
        if page is None:
            return 0, []
        return page["meta"]["count"], [to_work(r) for r in page["results"]]

    def _pace(self) -> None:
        if self._last_request_at is not None:
            elapsed = self._clock() - self._last_request_at
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                self._sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_at = self._clock()

    def _request(self, path: str, params: dict) -> dict | None:
        if self._mailto:
            params = {**params, "mailto": self._mailto}
        problem = ""
        for attempt in range(1, self._max_attempts + 1):
            wait = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            self._pace()
            try:
                response = self._http.get(OPENALEX_API + path, params=params)
            except httpx.TransportError as exc:
                problem = f"network error ({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 404:
                    return None
                if response.status_code not in RETRYABLE_STATUSES:
                    raise OpenAlexError(
                        f"OpenAlex rejected the request: HTTP {response.status_code}"
                    )
                problem = f"HTTP {response.status_code}"
            if attempt < self._max_attempts:
                logger.warning(
                    "OpenAlex %s on attempt %d/%d; retrying in %.0f s",
                    problem,
                    attempt,
                    self._max_attempts,
                    wait,
                )
                self._sleep(wait)
        raise OpenAlexUnavailableError(
            f"OpenAlex unavailable after {self._max_attempts} attempts: {problem}"
        )
