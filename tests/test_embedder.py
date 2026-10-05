import numpy as np
import pytest

from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder("BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX)


def test_describes_itself(embedder):
    assert embedder.dimension == 384
    assert embedder.max_tokens == 512


def test_count_tokens_includes_special_tokens(embedder):
    assert embedder.count_tokens("LLM judges prefer the first answer.") == 10


def test_passages_are_normalized_vectors(embedder):
    vectors = embedder.embed_passages(["first passage", "second passage"])

    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_query_finds_the_relevant_passage(embedder):
    passages = [
        "The judge sentenced the defendant to five years in prison.",
        "LLM evaluators tend to prefer whichever answer is shown first.",
        "Preheat the oven to 200 degrees before baking the bread.",
    ]

    scores = embedder.embed_passages(passages) @ embedder.embed_query(
        "Which answer do language-model judges favor?"
    )

    assert int(np.argmax(scores)) == 1


def test_rejects_text_over_the_limit(embedder):
    with pytest.raises(ValueError, match="exceed 512 tokens"):
        embedder.embed_passages(["word " * 600])
