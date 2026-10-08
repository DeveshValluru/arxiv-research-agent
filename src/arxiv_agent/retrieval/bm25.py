import math
import re
import threading
from collections import Counter

import snowballstemmer

WORD = re.compile(r"[a-z0-9]+")
# A Snowball stemmer keeps its working state on the object, so one shared by
# threads corrupts words: concurrent eval questions (7.2) got IndexErrors and,
# worse, silently wrong stems. One stemmer per thread.
_local = threading.local()
# Function words carry no topic. Question words matter more than usual: papers
# rarely contain "does" or "which", so BM25 would give them a high IDF.
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
    }
)


def tokenize(text: str) -> list[str]:
    words = [word for word in WORD.findall(text.lower()) if word not in STOPWORDS]
    if not hasattr(_local, "stemmer"):
        _local.stemmer = snowballstemmer.stemmer("english")
    return _local.stemmer.stemWords(words)


class BM25Index:
    # Built in memory per paper: a few hundred chunks take milliseconds, and
    # IDF is then measured within the paper the question is about.
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(document)) for document in documents]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        average = sum(self._lengths) / len(self._docs) if self._docs else 0.0
        self._avg_length = average or 1.0  # every document empty after tokenizing
        n = len(self._docs)
        doc_freq = Counter(term for doc in self._docs for term in doc)
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in doc_freq.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = list(dict.fromkeys(tokenize(query)))
        scores = []
        for doc, length in zip(self._docs, self._lengths):
            # Long documents match words by chance more often: raise the bar.
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_length)
            scores.append(
                sum(
                    self._idf[term] * doc[term] * (self.k1 + 1) / (doc[term] + norm)
                    for term in terms
                    if doc[term]
                )
            )
        return scores

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        # Document positions and scores, best first. Documents that share no
        # word with the query are left out rather than returned with score 0.
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]
