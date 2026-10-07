from sentence_transformers import CrossEncoder

from arxiv_agent.ingestion.models import Chunk

# Chosen by benchmark (3.4): the 22M model got most of the gain at 0.65 s per
# question on CPU; bge-reranker-v2-m3 scored higher but took 18.6 s.
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def rerank_text(chunk: Chunk) -> str:
    # The heading restores context the chunker cut away ("this" -> position
    # bias). Measured: R@5 0.91 with it, 0.85 without.
    return f"{' > '.join(chunk.section_path)}\n{chunk.text}"


class Reranker:
    # A cross-encoder reads the question and each chunk together, so it ranks
    # more precisely than comparing two separate vectors, but every pair costs
    # a full model run. Use it on a short candidate list, never a whole paper.
    def __init__(
        self, model_id: str = DEFAULT_RERANKER, model: CrossEncoder | None = None
    ) -> None:
        self.model_id = model_id
        self._model = model or CrossEncoder(model_id, max_length=512)

    def rerank(
        self, question: str, chunks: list[Chunk], top_k: int | None = None
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        scores = self._model.predict([(question, rerank_text(c)) for c in chunks])
        # sorted() is stable: chunks with equal scores keep their retrieval order
        ranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
