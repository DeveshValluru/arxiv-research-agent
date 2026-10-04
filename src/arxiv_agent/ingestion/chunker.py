import math
import re
from arxiv_agent.ingestion.models import Section

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
ABBREVIATIONS = (
    "et al.",
    "e.g.",
    "i.e.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Sec.",
    "Tab.",
    "vs.",
    "cf.",
    "etc.",
)


def _approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _section_paths(sections: list[Section]) -> list[list[str]]:
    paths = []
    path: list[str] = []

    for section in sections:
        path = path[: section.level - 1] + [section.title]
        paths.append(path)

    return paths


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in SENTENCE_END.split(text):
        if sentences and sentences[-1].endswith(ABBREVIATIONS):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)

    return sentences
