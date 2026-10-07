from arxiv_agent.evals.retrieval import match_evidence

CHUNKS = [
    "Introduction. Judges are studied widely.",
    (
        "We created a corpus of human-human anti-scam dialogs (see Table 1) "
        "to learn human elicitation strategies. Each dialog has $\\alpha$ turns."
    ),
    "Results. The model improves persuasion success by a wide margin overall.",
]


def test_match_ignores_punctuation_dashes_and_spacing():
    evidence = [
        (
            "We created a corpus of human–human anti‑scam dialogs, "
            "see Table 1, to learn human elicitation strategies."
        )
    ]
    assert match_evidence(CHUNKS, evidence) == {1}


def test_one_matching_sentence_is_enough():
    evidence = [
        (
            "Each dialog in our corpus has exactly α turns. "
            "The model improves persuasion success by a wide margin overall."
        )
    ]
    assert match_evidence(CHUNKS, evidence) == {2}


def test_short_sentences_are_too_generic_to_match():
    assert match_evidence(CHUNKS, ["Judges are studied widely."]) == set()


def test_words_must_match_whole_words():
    evidence = ["e created a corpus of human human anti scam dialog"]
    assert match_evidence(CHUNKS, evidence) == set()


def test_unmatched_evidence_gives_an_empty_set():
    evidence = ["This sentence appears nowhere in the paper at all, not even once."]
    assert match_evidence(CHUNKS, evidence) == set()
