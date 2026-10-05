import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict

from arxiv_agent.clients.arxiv import PaperSummary
from arxiv_agent.ingestion.models import Chunk

# Three layers, each tagged with what produced it:
#   papers  <- arXiv metadata, one row per (arxiv_id, version)
#   chunks  <- chunk_paper(); chunker_version says which recipe made them
#   chunk_vectors_<model>  <- one table per embedder, because a pgvector column
#                             has a fixed dimension (384 for bge-small, 1024 for Qwen3)
# Deleting a paper's chunks cascades to every vector table.
SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id         text        NOT NULL,
    version          integer     NOT NULL,
    title            text        NOT NULL,
    authors          text[]      NOT NULL,
    abstract         text        NOT NULL,
    published        timestamptz NOT NULL,
    updated          timestamptz NOT NULL,
    primary_category text        NOT NULL,
    categories       text[]      NOT NULL,
    ingested_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (arxiv_id, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        text    PRIMARY KEY,
    arxiv_id        text    NOT NULL,
    version         integer NOT NULL,
    chunk_index     integer NOT NULL,
    kind            text    NOT NULL CHECK (kind IN ('abstract', 'text', 'table')),
    section_path    text[]  NOT NULL,
    text            text    NOT NULL,
    embed_text      text    NOT NULL,
    token_count     integer NOT NULL,
    chunker_version integer NOT NULL,
    text_search     tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (arxiv_id, version, chunk_index),
    FOREIGN KEY (arxiv_id, version) REFERENCES papers ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chunks_text_search_idx ON chunks USING gin (text_search);
"""

PAPER_COLUMNS = sql.SQL(
    "arxiv_id, version, title, authors, abstract, published, updated, "
    "primary_category, categories"
)
CHUNK_COLUMNS = sql.SQL(
    "c.chunk_id, c.arxiv_id, c.version, c.chunk_index AS index, c.kind, "
    "c.section_path, c.text, c.embed_text, c.token_count, c.chunker_version"
)
MAX_IDENTIFIER_LENGTH = 63


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float


def vector_table(model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    name = f"chunk_vectors_{slug}"
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"table name for {model_id!r} is too long: {name}")
    return name


def _paper_scope(papers: list[tuple[str, int]] | None) -> tuple[sql.Composable, dict]:
    if papers is None:
        return sql.SQL("TRUE"), {}
    scope = sql.SQL(
        "(c.arxiv_id, c.version) IN "
        "(SELECT * FROM unnest(%(scope_ids)s::text[], %(scope_versions)s::int[]))"
    )
    return scope, {
        "scope_ids": [arxiv_id for arxiv_id, _ in papers],
        "scope_versions": [version for _, version in papers],
    }


def _hit(row: dict) -> SearchHit:
    score = row.pop("score")
    return SearchHit(chunk=Chunk.model_validate(row), score=score)


class ChunkStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    @classmethod
    def connect(cls, url: str) -> "ChunkStore":
        store = cls(psycopg.connect(url, autocommit=True, row_factory=dict_row))
        store.init_schema()
        return store

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        register_vector(self._conn)
        self._conn.execute(SCHEMA)

    def save_paper(self, paper: PaperSummary) -> None:
        self._conn.execute(
            """
            INSERT INTO papers (arxiv_id, version, title, authors, abstract,
                                published, updated, primary_category, categories)
            VALUES (%(arxiv_id)s, %(version)s, %(title)s, %(authors)s, %(abstract)s,
                    %(published)s, %(updated)s, %(primary_category)s, %(categories)s)
            ON CONFLICT (arxiv_id, version) DO UPDATE SET
                title = EXCLUDED.title,
                authors = EXCLUDED.authors,
                abstract = EXCLUDED.abstract,
                published = EXCLUDED.published,
                updated = EXCLUDED.updated,
                primary_category = EXCLUDED.primary_category,
                categories = EXCLUDED.categories
            """,
            paper.model_dump(),
        )

    def get_paper(self, arxiv_id: str, version: int) -> PaperSummary | None:
        row = self._conn.execute(
            sql.SQL(
                "SELECT {} FROM papers WHERE arxiv_id = %s AND version = %s"
            ).format(PAPER_COLUMNS),
            (arxiv_id, version),
        ).fetchone()
        return PaperSummary.model_validate(row) if row else None

    def stored_chunker_version(self, arxiv_id: str, version: int) -> int | None:
        row = self._conn.execute(
            "SELECT min(chunker_version) AS chunker_version FROM chunks "
            "WHERE arxiv_id = %s AND version = %s",
            (arxiv_id, version),
        ).fetchone()
        return row["chunker_version"]

    def replace_chunks(self, arxiv_id: str, version: int, chunks: list[Chunk]) -> None:
        strays = [
            c.chunk_id for c in chunks if (c.arxiv_id, c.version) != (arxiv_id, version)
        ]
        if strays:
            raise ValueError(f"chunks from another paper: {strays[:3]}")

        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE arxiv_id = %s AND version = %s",
                (arxiv_id, version),
            )
            cur.executemany(
                """
                INSERT INTO chunks (chunk_id, arxiv_id, version, chunk_index, kind,
                                    section_path, text, embed_text, token_count,
                                    chunker_version)
                VALUES (%(chunk_id)s, %(arxiv_id)s, %(version)s, %(index)s, %(kind)s,
                        %(section_path)s, %(text)s, %(embed_text)s, %(token_count)s,
                        %(chunker_version)s)
                """,
                [chunk.model_dump() for chunk in chunks],
            )

    def get_chunks(self, arxiv_id: str, version: int) -> list[Chunk]:
        rows = self._conn.execute(
            sql.SQL(
                "SELECT {} FROM chunks c WHERE c.arxiv_id = %s AND c.version = %s "
                "ORDER BY c.chunk_index"
            ).format(CHUNK_COLUMNS),
            (arxiv_id, version),
        ).fetchall()
        return [Chunk.model_validate(row) for row in rows]

    def ensure_vector_table(self, model_id: str, dimension: int) -> None:
        self._conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    chunk_id  text PRIMARY KEY REFERENCES chunks ON DELETE CASCADE,
                    embedding vector({}) NOT NULL
                )
                """
            ).format(sql.Identifier(vector_table(model_id)), sql.Literal(dimension))
        )

    def chunks_without_vectors(
        self, arxiv_id: str, version: int, model_id: str
    ) -> list[Chunk]:
        rows = self._conn.execute(
            sql.SQL(
                """
                SELECT {} FROM chunks c
                WHERE c.arxiv_id = %s AND c.version = %s
                  AND NOT EXISTS (SELECT 1 FROM {} v WHERE v.chunk_id = c.chunk_id)
                ORDER BY c.chunk_index
                """
            ).format(CHUNK_COLUMNS, sql.Identifier(vector_table(model_id))),
            (arxiv_id, version),
        ).fetchall()
        return [Chunk.model_validate(row) for row in rows]

    def save_vectors(
        self, model_id: str, chunk_ids: list[str], vectors: np.ndarray
    ) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError(f"{len(chunk_ids)} chunk ids but {len(vectors)} vectors")

        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {} (chunk_id, embedding) VALUES (%s, %s) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding"
                ).format(sql.Identifier(vector_table(model_id))),
                list(zip(chunk_ids, vectors)),
            )

    def vector_search(
        self,
        model_id: str,
        query_vector: np.ndarray,
        k: int = 10,
        papers: list[tuple[str, int]] | None = None,
    ) -> list[SearchHit]:
        # <#> is pgvector's negative inner product; on normalized vectors the
        # inner product is the cosine similarity, so score = -(a <#> b).
        # No ANN index yet: this is an exact scan, the same ranking as numpy.
        scope, params = _paper_scope(papers)
        rows = self._conn.execute(
            sql.SQL(
                """
                SELECT {columns}, -(v.embedding <#> %(query)s) AS score
                FROM {table} v JOIN chunks c USING (chunk_id)
                WHERE {scope}
                ORDER BY v.embedding <#> %(query)s, c.chunk_id
                LIMIT %(k)s
                """
            ).format(
                columns=CHUNK_COLUMNS,
                table=sql.Identifier(vector_table(model_id)),
                scope=scope,
            ),
            {"query": query_vector, "k": k, **params},
        ).fetchall()
        return [_hit(row) for row in rows]

    def keyword_search(
        self,
        query: str,
        k: int = 10,
        papers: list[tuple[str, int]] | None = None,
    ) -> list[SearchHit]:
        # plainto_tsquery stems the words and drops stopwords, but ANDs them,
        # so a full question rarely matches anything. Swapping & for | lets
        # any word match, and ts_rank_cd rewards chunks that match more words.
        scope, params = _paper_scope(papers)
        rows = self._conn.execute(
            sql.SQL(
                """
                SELECT {columns}, ts_rank_cd(c.text_search, q.tsq) AS score
                FROM chunks c,
                     (SELECT replace(plainto_tsquery('english', %(query)s)::text,
                                     ' & ', ' | ')::tsquery AS tsq) q
                WHERE c.text_search @@ q.tsq AND {scope}
                ORDER BY score DESC, c.chunk_id
                LIMIT %(k)s
                """
            ).format(columns=CHUNK_COLUMNS, scope=scope),
            {"query": query, "k": k, **params},
        ).fetchall()
        return [_hit(row) for row in rows]
