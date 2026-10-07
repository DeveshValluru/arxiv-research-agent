import re
from collections import Counter

from arxiv_agent.evals.retrieval import WORD, gold_indices, match_evidence
from arxiv_agent.ingestion.models import Chunk

CITATION_MARK = re.compile(r"\[S\d+\]")
ARTICLES = {"a", "an", "the"}


def answer_tokens(text: str) -> list[str]:
    # The SQuAD/QASPER convention: lowercase words, no punctuation, no articles.
    # Our [S1] citation marks are removed first: they aren't part of the answer.
    text = CITATION_MARK.sub(" ", text).lower()
    return [word for word in WORD.findall(text) if word not in ARTICLES]


def token_f1(prediction: str, gold: str) -> float:
    predicted, expected = answer_tokens(prediction), answer_tokens(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, gold_answers: list[str]) -> float:
    # Annotators phrase the same answer differently: score against the closest.
    return max((token_f1(prediction, gold) for gold in gold_answers), default=0.0)


def refusal_correct(item_type: str, status: str) -> bool | None:
    # Deterministic, thanks to the exact refusal sentence. False premises need
    # the judge: refusing one is safe but not the best answer.
    if item_type == "unanswerable":
        return status == "refused"
    if item_type == "answerable":
        return status != "refused"
    return None


def gold_chunk_ids(chunks: list[Chunk], evidence: list[str]) -> set[str]:
    # Short phrases (our survey labels) match exactly; long paragraphs from
    # another parser (QASPER) match sentence by sentence. Either one counts.
    texts = [chunk.text for chunk in chunks]
    found = gold_indices(texts, evidence) | match_evidence(texts, evidence)
    return {chunks[i].chunk_id for i in found}


def evidence_hit(gold_ids: set[str], retrieved_ids: list[str]) -> bool | None:
    # None means "we can't tell": the evidence never matched our chunks, so this
    # question must not count as a retrieval miss.
    if not gold_ids:
        return None
    return any(chunk_id in gold_ids for chunk_id in retrieved_ids)
