from pydantic import BaseModel, ConfigDict


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    level: int
    text: str


class ParsedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    abstract: str
    sections: list[Section]
