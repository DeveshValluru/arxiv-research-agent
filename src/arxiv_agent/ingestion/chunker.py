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
# 4: table headers, space-less words and chunk headers capped (found by the
# 5.4 eval: chunks over the embedder's 512-token limit)
CHUNKER_VERSION = 4
HEADER_TOKENS = 100  # title and section path, in front of each embedded chunk
# Stored chunks are shared by every embedder, so they're always measured with this
# one tokenizer. CHUNKER_VERSION covers the algorithm and these settings: bump it
# when MAX_TOKENS or CHUNK_TOKENIZER changes, so stored chunks get rebuilt.
CHUNK_TOKENIZER = "BAAI/bge-small-en-v1.5"


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


def _fit(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    # A piece with no spaces left to split at (a long formula, a URL): halve it
    # by characters until each part fits. A single character always does.
    if count_tokens(text) <= max_tokens:
        return [text]
    middle = len(text) // 2
    return _fit(text[:middle], max_tokens, count_tokens) + _fit(
        text[middle:], max_tokens, count_tokens
    )


def _truncate(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> str:
    # The longest run of leading words that fits, marked as cut.
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split(" ")
    low, high = 0, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(" ".join(words[:middle]) + " …") <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return " ".join(words[:low]) + " …"


def _sentence_units(
    paragraph: str, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[str]:
    units: list[str] = []
    for sentence in _split_sentences(paragraph):
        if count_tokens(sentence) <= max_tokens:
            units.append(sentence)
        else:
            for word in sentence.split():
                units.extend(_fit(word, max_tokens, count_tokens))
    return units


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
            sentences = _sentence_units(paragraph, max_tokens, count_tokens)
            chunks.extend(_pack(sentences, max_tokens, " ", count_tokens))

    chunks.extend(_pack(small, max_tokens, "\n\n", count_tokens))
    return chunks


def _with_header(
    title: str, path: list[str], body: str, count_tokens: Callable[[str], int]
) -> str:
    # Capped, so a body of max_tokens plus its header always fits the embedder.
    header = _truncate(f"{title}\n{' > '.join(path)}", HEADER_TOKENS, count_tokens)
    return f"{header}\n\n{body}"


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
                    _with_header(paper.title, ["Abstract"], text, count_tokens),
                )
            )

    for section, path in zip(paper.sections, _section_paths(paper.sections)):
        if not section.text:
            continue

        for text in _chunk_texts(section.text, max_tokens, count_tokens):
            embed_text = _with_header(paper.title, path, text, count_tokens)
            pieces.append(("text", path, text, embed_text))

    for table in paper.tables:
        path = [table.section] if table.section else []
        columns = table.markdown.split("\n")[0]
        text = f"{table.caption}\n\n{table.markdown}"
        # Capped: some papers put whole prompts in a "table", and its header
        # row ran to 987 tokens, past what the embedder takes (an eval found it).
        description = _truncate(
            f"{table.caption}\nColumns: {columns}", max_tokens, count_tokens
        )
        pieces.append(
            (
                "table",
                path,
                text,
                _with_header(paper.title, path, description, count_tokens),
            )
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
