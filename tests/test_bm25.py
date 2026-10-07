import pytest

from arxiv_agent.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_drops_stopwords_and_stems():
    assert tokenize("Which base model does PandaLM fine-tune?") == [
        "base",
        "model",
        "pandalm",
        "fine",
        "tune",
    ]


def test_word_forms_share_a_stem():
    assert tokenize("tunes tuned tuning") == ["tune", "tune", "tune"]


def test_stemming_lets_tune_match_tunes():
    index = BM25Index(["PandaLM fine-tunes LLaMA-7B.", "Unrelated text here."])
    assert index.top("fine-tune", k=5)[0][0] == 0


def test_rare_word_outweighs_common_ones():
    docs = [
        "judge model judge model judge",
        "judge model judge model judge model judge",
        "world model judge",
        "judge model",
    ]
    assert BM25Index(docs).top("world model judge", k=1)[0][0] == 2


def test_repeating_a_word_saturates():
    index = BM25Index(["bias", "bias " * 10, "other words entirely"])
    once, ten_times, _ = index.scores("bias")
    assert once < ten_times < once * 2.2  # k1 + 1 = 2.2 is the ceiling


def test_longer_documents_need_more_to_score_as_high():
    index = BM25Index(["bias here", "bias " + "filler " * 20])
    short, long = index.scores("bias")
    assert short > long


def test_top_sorts_cuts_at_k_and_drops_non_matches():
    index = BM25Index(["alpha", "alpha alpha beta", "gamma", "beta"])
    top = index.top("alpha beta", k=2)
    assert [i for i, _ in top] == [1, 0]
    assert all(score > 0 for _, score in top)
    assert index.top("gamma", k=10) == [(2, pytest.approx(index.scores("gamma")[2]))]


def test_question_made_only_of_stopwords_matches_nothing():
    assert BM25Index(["what does it do"]).top("What does it do?", k=5) == []


def test_empty_index_is_harmless():
    assert BM25Index([]).top("anything", k=5) == []
