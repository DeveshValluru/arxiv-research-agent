from typing import Literal, Protocol

from arxiv_agent.ingestion.embedder import Embedder, HostedEmbedder
from arxiv_agent.ingestion.models import Chunk
from arxiv_agent.retrieval.bm25 import BM25Index
from arxiv_agent.retrieval.fusion import reciprocal_rank_fusion
from arxiv_agent.retrieval.reranker import Reranker
from arxiv_agent.storage.chunk_store import ChunkStore, SearchHit

RetrieverKind = Literal["dense", "hybrid", "rerank"]
CANDIDATES = 20  # how many chunks each first-stage method passes on


class Retriever(Protocol):
    # Anything with a name and a retrieve() method can feed the Answerer.
    # Scores are only comparable within one retriever (cosine, RRF, or
    # cross-encoder), so callers should use the order, not the numbers.
    name: str

    def retrieve(
        self, question: str, paper: tuple[str, int], k: int
    ) -> list[SearchHit]: ...


class DenseRetriever:
    def __init__(self, store: ChunkStore, embedder: Embedder | HostedEmbedder) -> None:
        self._store = store
        self._embedder = embedder
        self.name = f"dense({embedder.model_id})"

    def retrieve(
        self, question: str, paper: tuple[str, int], k: int
    ) -> list[SearchHit]:
        return self._store.vector_search(
            self._embedder.model_id,
            self._embedder.embed_query(question),
            k=k,
            papers=[paper],
        )


class HybridRetriever:
    # Retrieve for recall, rerank for precision: dense and BM25 each pass on
    # their top CANDIDATES, RRF fuses the two rankings, and an optional
    # cross-encoder re-reads the fused list to put the best chunks first.
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder | HostedEmbedder,
        reranker: Reranker | None = None,
        candidates: int = CANDIDATES,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates
        self._papers: dict[tuple[str, int], tuple[dict[str, Chunk], BM25Index]] = {}
        self.name = f"hybrid({embedder.model_id}+bm25)" + (
            f"+rerank({reranker.model_id})" if reranker else ""
        )

    def _paper(self, paper: tuple[str, int]) -> tuple[dict[str, Chunk], BM25Index]:
        # Built once per paper and kept for this retriever's lifetime. A paper
        # re-ingested meanwhile needs a new retriever (fine for scripts; the
        # API will need invalidation).
        if paper not in self._papers:
            chunks = self._store.get_chunks(*paper)
            self._papers[paper] = (
                {chunk.chunk_id: chunk for chunk in chunks},
                BM25Index([chunk.text for chunk in chunks]),
            )
        return self._papers[paper]

    def retrieve(
        self, question: str, paper: tuple[str, int], k: int
    ) -> list[SearchHit]:
        by_id, bm25 = self._paper(paper)
        ids = list(by_id)
        dense = self._store.vector_search(
            self._embedder.model_id,
            self._embedder.embed_query(question),
            k=self._candidates,
            papers=[paper],
        )
        fused = reciprocal_rank_fusion(
            [
                [hit.chunk.chunk_id for hit in dense],
                [ids[i] for i, _ in bm25.top(question, self._candidates)],
            ]
        )[: self._candidates]

        if self._reranker is None:
            return [
                SearchHit(chunk=by_id[cid], score=score) for cid, score in fused[:k]
            ]
        reranked = self._reranker.rerank(
            question, [by_id[cid] for cid, _ in fused], top_k=k
        )
        return [SearchHit(chunk=chunk, score=score) for chunk, score in reranked]


def build_retriever(
    kind: RetrieverKind, store: ChunkStore, embedder: Embedder | HostedEmbedder
) -> Retriever:
    if kind == "dense":
        return DenseRetriever(store, embedder)
    if kind == "hybrid":
        return HybridRetriever(store, embedder)
    return HybridRetriever(store, embedder, reranker=Reranker())
