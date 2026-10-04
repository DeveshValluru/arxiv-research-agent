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
