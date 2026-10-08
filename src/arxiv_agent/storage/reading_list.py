"""The reading list: papers saved from search results or reviews (8.1).

One list (a local demo has one user). A paper is saved with what the UI
shows, so the list renders without asking arXiv again.
"""

from datetime import datetime

import psycopg
from pydantic import BaseModel, ConfigDict

SCHEMA = """
CREATE TABLE IF NOT EXISTS reading_list (
    arxiv_id  text PRIMARY KEY,
    version   integer NOT NULL,
    title     text NOT NULL,
    authors   text[] NOT NULL,
    published text NOT NULL,
    note      text NOT NULL DEFAULT '',
    added_at  timestamptz NOT NULL DEFAULT now()
);
"""


class SavedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    published: str
    note: str = ""
    added_at: datetime | None = None


class ReadingList:
    def __init__(self, conn: psycopg.Connection) -> None:
        # conn: autocommit, dict rows (as the other stores use).
        self._conn = conn
        self._conn.execute(SCHEMA)

    def all(self) -> list[SavedPaper]:
        rows = self._conn.execute(
            "SELECT * FROM reading_list ORDER BY added_at DESC"
        ).fetchall()
        return [SavedPaper(**row) for row in rows]

    def save(self, paper: SavedPaper) -> SavedPaper:
        # Saving again updates the paper and its note, and keeps its place.
        row = self._conn.execute(
            """
            INSERT INTO reading_list (arxiv_id, version, title, authors, published, note)
            VALUES (%(arxiv_id)s, %(version)s, %(title)s, %(authors)s, %(published)s,
                    %(note)s)
            ON CONFLICT (arxiv_id) DO UPDATE SET
                version = EXCLUDED.version, title = EXCLUDED.title,
                authors = EXCLUDED.authors, published = EXCLUDED.published,
                note = EXCLUDED.note
            RETURNING *
            """,
            paper.model_dump(exclude={"added_at"}),
        ).fetchone()
        return SavedPaper(**row)

    def remove(self, arxiv_id: str) -> bool:
        row = self._conn.execute(
            "DELETE FROM reading_list WHERE arxiv_id = %s RETURNING 1", (arxiv_id,)
        ).fetchone()
        return row is not None

    def ids(self) -> set[str]:
        rows = self._conn.execute("SELECT arxiv_id FROM reading_list").fetchall()
        return {row["arxiv_id"] for row in rows}
