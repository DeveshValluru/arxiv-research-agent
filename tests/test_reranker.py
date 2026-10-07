from pathlib import Path

from arxiv_agent.ingestion.chunker import chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.retrieval.reranker import Reranker, rerank_text

FIXTURES = Path(__file__).parent / "fixtures"
CHUNKS = chunk_paper(
    parse_arxiv_html((FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")),
    "2499.00001",
    1,
)


class FakeCrossEncoder:
    # Scores a pair by how many question words appear in the passage.
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return [
            float(sum(word in passage.lower() for word in question.lower().split()))
            for question, passage in pairs
        ]


def make_reranker() -> tuple[Reranker, FakeCrossEncoder]:
    model = FakeCrossEncoder()
    return Reranker(model_id="test/fake-reranker", model=model), model


def test_rerank_text_puts_the_section_path_first():
    assert rerank_text(CHUNKS[4]) == (
        "2. Method > 2.1. Position Bias > 2.1.1. Swapping\n"
        "Swapping the order reduces the bias by $\\Delta b$."
    )


def test_rerank_sorts_by_score():
    reranker, _ = make_reranker()

    ranked = reranker.rerank("swapping order bias", CHUNKS)

    assert ranked[0][0] == CHUNKS[4]
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_the_model_sees_the_question_with_each_chunk():
    reranker, model = make_reranker()

    reranker.rerank("why swap?", CHUNKS[:2])

    assert model.pairs == [("why swap?", rerank_text(c)) for c in CHUNKS[:2]]


def test_top_k_cuts_the_list():
    reranker, _ = make_reranker()
    assert len(reranker.rerank("bias", CHUNKS, top_k=2)) == 2


def test_ties_keep_retrieval_order():
    reranker, _ = make_reranker()
    ranked = reranker.rerank("zzz", CHUNKS)  # no word matches: every score is 0
    assert [chunk for chunk, _ in ranked] == CHUNKS


def test_no_candidates_skips_the_model():
    reranker, model = make_reranker()
    assert reranker.rerank("anything", []) == []
    assert model.pairs == []
