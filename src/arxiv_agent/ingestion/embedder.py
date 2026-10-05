from collections.abc import Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"


def load_token_counter(model_id: str) -> Callable[[str], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, verbose=False)["input_ids"])

    return count_tokens


class Embedder:
    def __init__(
        self,
        model_id: str,
        query_prefix: str = "",
        revision: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self._model = SentenceTransformer(model_id, revision=revision)
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    @property
    def max_tokens(self) -> int:
        return self._model.max_seq_length

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text, verbose=False)["input_ids"])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        too_long = [
            i
            for i, text in enumerate(texts)
            if self.count_tokens(text) > self.max_tokens
        ]

        if too_long:
            raise ValueError(
                f"{len(too_long)} text(s) exceed {self.max_tokens} tokens "
                f"(first at index {too_long[0]}); chunk with count_tokens=embedder.count_tokens"
            )

        return self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self._model.encode(self._query_prefix + query, normalize_embeddings=True)
