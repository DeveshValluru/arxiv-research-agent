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


def _pack(units: list[str], max_tokens: int, separator: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = separator.join(current + [unit])

        if current and _approx_tokens(candidate) > max_tokens:
            chunks.append(separator.join(current))
            current = [unit]

        else:
            current.append(unit)

    if current:
        chunks.append(separator.join(current))
    return chunks


def _chunk_texts(text: str, max_tokens: int) -> list[str]:
    chunks: list[str] = []
    small: list[str] = []

    for paragraph in text.split("\n\n"):
        if _approx_tokens(paragraph) <= max_tokens:
            small.append(paragraph)
        else:
            chunks.extend(_pack(small, max_tokens, "\n\n"))
            small = []
            chunks.extend(_pack(_split_sentences(paragraph), max_tokens, " "))

    chunks.extend(_pack(small, max_tokens, "\n\n"))
    return chunks
