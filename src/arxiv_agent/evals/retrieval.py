import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-z0-9]+")
MIN_EVIDENCE_WORDS = 8
QUOTE_FIXES = {"‘": "'", "’": "'", "“": '"', "”": '"'}


def normalize(text: str) -> str:
    for curly, straight in QUOTE_FIXES.items():
        text = text.replace(curly, straight)
    return " ".join(text.lower().split())


def gold_indices(chunk_texts: list[str], evidence: list[str]) -> set[int]:
    phrases = [normalize(phrase) for phrase in evidence]
    return {
        i
        for i, text in enumerate(chunk_texts)
        if any(phrase in normalize(text) for phrase in phrases)
    }


def first_hit_rank(ranking: list[int], gold: set[int]) -> int | None:
    for position, chunk_index in enumerate(ranking, start=1):
        if chunk_index in gold:
            return position
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    return sum(1 / rank if rank else 0.0 for rank in ranks) / len(ranks)


def _words(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def match_evidence(chunk_texts: list[str], paragraphs: list[str]) -> set[int]:
    chunks = [f" {_words(text)} " for text in chunk_texts]

    sentences = [
        f" {words} "
        for paragraph in paragraphs
        for sentence in SENTENCE_END.split(paragraph)
        if len((words := _words(sentence)).split()) >= MIN_EVIDENCE_WORDS
    ]

    return {i for i, chunk in enumerate(chunks) if any(s in chunk for s in sentences)}
