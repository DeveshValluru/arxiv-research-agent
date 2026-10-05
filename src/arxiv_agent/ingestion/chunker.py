import math
import re
from collections.abc import Callable

from arxiv_agent.ingestion.models import Chunk, ParsedPaper, Section

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
MAX_TOKENS = 350
CHUNKER_VERSION = 2


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


def _pack(
    units: list[str],
    max_tokens: int,
    separator: str,
    count_tokens: Callable[[str], int] = _approx_tokens,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = separator.join(current + [unit])

        if current and count_tokens(candidate) > max_tokens:
            chunks.append(separator.join(current))
            current = [unit]

        else:
            current.append(unit)

    if current:
        chunks.append(separator.join(current))
    return chunks


def _chunk_texts(
    text: str,
    max_tokens: int,
    count_tokens: Callable[[str], int] = _approx_tokens,
) -> list[str]:
    chunks: list[str] = []
    small: list[str] = []

    for paragraph in text.split("\n\n"):
        if count_tokens(paragraph) <= max_tokens:
            small.append(paragraph)
        else:
            chunks.extend(_pack(small, max_tokens, "\n\n", count_tokens))
            small = []
            sentences = _split_sentences(paragraph)
            chunks.extend(_pack(sentences, max_tokens, " ", count_tokens))

    chunks.extend(_pack(small, max_tokens, "\n\n", count_tokens))
    return chunks


def _with_header(title: str, path: list[str], body: str) -> str:
    return f"{title}\n{' > '.join(path)}\n\n{body}"


def chunk_paper(
    paper: ParsedPaper,
    arxiv_id: str,
    version: int,
    max_tokens: int = MAX_TOKENS,
    count_tokens: Callable[[str], int] = _approx_tokens,
) -> list[Chunk]:
    pieces: list[tuple[str, list[str], str, str]] = []

    if paper.abstract:
        for text in _chunk_texts(paper.abstract, max_tokens, count_tokens):
            pieces.append(
                (
                    "abstract",
                    ["Abstract"],
                    text,
                    _with_header(paper.title, ["Abstract"], text),
                )
            )

    for section, path in zip(paper.sections, _section_paths(paper.sections)):
        if not section.text:
            continue

        for text in _chunk_texts(section.text, max_tokens, count_tokens):
            pieces.append(("text", path, text, _with_header(paper.title, path, text)))

    for table in paper.tables:
        path = [table.section] if table.section else []
        columns = table.markdown.split("\n")[0]
        text = f"{table.caption}\n\n{table.markdown}"
        description = f"{table.caption}\nColumns: {columns}"
        pieces.append(
            ("table", path, text, _with_header(paper.title, path, description))
        )

    return [
        Chunk(
            chunk_id=f"{arxiv_id}v{version}:{index:04d}",
            arxiv_id=arxiv_id,
            version=version,
            index=index,
            kind=kind,
            section_path=path,
            text=text,
            embed_text=embed_text,
            token_count=count_tokens(embed_text),
            chunker_version=CHUNKER_VERSION,
        )
        for index, (kind, path, text, embed_text) in enumerate(pieces)
    ]
