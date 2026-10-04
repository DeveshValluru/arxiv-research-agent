from typing import Literal

from pydantic import BaseModel, ConfigDict


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    level: int
    text: str


class Table(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: str
    caption: str
    markdown: str


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_id: str
    text: str
    arxiv_id: str | None = None


class ParsedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    abstract: str
    sections: list[Section]
    tables: list[Table]
    references: list[Reference]
    hidden_text: list[str]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    arxiv_id: str
    version: int
    index: int
    kind: Literal["abstract", "text", "table"]
    section_path: list[str]
    text: str
    embed_text: str
    token_count: int
    chunker_version: int
