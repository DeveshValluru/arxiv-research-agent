import numpy as np
import pytest

from arxiv_agent.ingestion.embedder import HostedEmbedder


class FakeClient:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def feature_extraction(self, text, model=None, **kwargs):
        batch = [text] if isinstance(text, str) else list(text)
        self.calls.append(batch)
        padding = [0.0] * (self.dimension - 2)
        return np.array([[len(t), 1.0, *padding] for t in batch], dtype=np.float32)


def make(client: FakeClient, **kwargs) -> HostedEmbedder:
    return HostedEmbedder(
        "test/hosted", dimension=3, provider="fake", client=client, **kwargs
    )


def test_vectors_are_normalized_even_if_the_provider_does_not():
    vectors = make(FakeClient()).embed_passages(["short", "a much longer passage"])

    assert vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_passages_are_sent_in_batches():
    client = FakeClient()
    make(client, batch_size=2).embed_passages(["a", "b", "c", "d", "e"])

    assert [len(call) for call in client.calls] == [2, 2, 1]


def test_query_gets_the_prefix_but_passages_do_not():
    client = FakeClient()
    embedder = make(client, query_prefix="Query: ")

    embedder.embed_passages(["passage"])
    vector = embedder.embed_query("question")

    assert client.calls == [["passage"], ["Query: question"]]
    assert vector.shape == (3,)


def test_wrong_dimension_fails_loudly():
    with pytest.raises(ValueError, match="dimension"):
        make(FakeClient(dimension=5)).embed_passages(["text"])
