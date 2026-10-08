from arxiv_agent.evals.flywheel import (
    critic_candidates,
    judge_case,
    label_candidate,
    load_candidates,
    merge,
    qa_candidate,
    qa_item,
    review_case,
    save_candidates,
)

TRACE = {
    "trace_id": "t1",
    "time": "2026-10-07T21:00:00Z",
    "question": "Why swap the order?",
    "paper": "2411.15594v6",
    "answer": "Swapping reduces bias [S1].",
    "status": "answered",
    "problems": [""],
    "failed": [],
}
RESULT = {
    "question": "How biased are LLM judges?",
    "removed": ["All judges prefer the first answer."],
    "critique": {
        "checks": [
            {
                "sentence": "All judges prefer the first answer [K2].",
                "labels": ["K2"],
                "verdict": "overstated",
                "reason": "The passage tested one judge.",
            }
        ]
    },
    "claims": [
        {"label": "K2", "arxiv_id": "2406.07791", "passage": "GPT-4 prefers..."},
        {"label": "K3", "arxiv_id": "2306.05685", "passage": "Unrelated."},
    ],
}


def label_row(arxiv_id: str, label: str) -> dict:
    return {
        "job_id": "9718091f-b135-41ed-bb48-212c9fd7d649",
        "arxiv_id": arxiv_id,
        "label": label,
        "question": "How biased are LLMs used as judges?",
        "title": "A paper",
        "created_at": "2026-10-07 22:18:31+00:00",
    }


def test_only_answers_a_guard_flagged_become_candidates():
    assert qa_candidate(TRACE) is None

    failed = qa_candidate(TRACE | {"failed": ["All judges agree [S1]."]})
    invalid = qa_candidate(TRACE | {"status": "invalid", "problems": ["no citations"]})

    assert (failed.id, failed.arxiv_id, failed.version) == ("qa:t1", "2411.15594", 6)
    assert failed.reason == "support check failed 1 sentence(s)"
    assert invalid.reason == "invalid answer: no citations"


def test_a_removed_sentence_comes_with_the_critics_verdict_and_cited_passages():
    [candidate] = critic_candidates("job1", "2026-10-07 21:00", "t9", RESULT)

    assert candidate.reason == "overstated: The passage tested one judge."
    assert (candidate.sources, candidate.passages) == (
        ["2406.07791"],
        ["GPT-4 prefers..."],
    )
    assert candidate.found == "2026-10-07"


def test_removals_without_a_judge_verdict_are_skipped():
    unchecked = RESULT | {"removed": ["A sentence nobody checked."]}
    uncited = RESULT | {
        "critique": {
            "checks": [
                RESULT["critique"]["checks"][0]
                | {"verdict": "uncited", "labels": [], "reason": "cites no claim"}
            ]
        }
    }

    assert critic_candidates("job1", "2026-10-07", None, unchecked) == []
    assert critic_candidates("job1", "2026-10-07", None, uncited) == []


def test_harvesting_again_keeps_what_was_triaged(tmp_path):
    first = label_candidate(label_row("2306.05685", "search_miss"))
    first.status = "rejected"
    path = tmp_path / "candidates.jsonl"
    save_candidates([first], path)

    merged = merge(
        load_candidates(path),
        [
            label_candidate(label_row("2306.05685", "search_miss")),
            label_candidate(label_row("2410.00001", "screener_false_positive")),
        ],
    )

    assert [(c.arxiv_id, c.status) for c in merged] == [
        ("2306.05685", "rejected"),
        ("2410.00001", "pending"),
    ]


def test_accepted_flags_become_eval_records():
    flagged = qa_candidate(TRACE | {"failed": ["x"]})
    item = qa_item(flagged, "answerable", ["It reduces position bias."])
    assert (item.source, item.type, item.arxiv_id, item.version) == (
        "flagged",
        "answerable",
        "2411.15594",
        6,
    )
    assert item.id == qa_item(flagged, "unanswerable", []).id  # stable

    [removed] = critic_candidates("job1", "2026-10-07", None, RESULT)
    case = judge_case(removed, supported=False)
    assert (case.kind, case.supported, case.sentence) == (
        "flagged",
        False,
        "All judges prefer the first answer.",
    )


def test_a_reviews_labels_become_one_review_case():
    labels = [
        label_candidate(label_row("2306.05685", "search_miss")),
        label_candidate(label_row("2410.00001", "screener_false_negative")),
        label_candidate(label_row("2501.00002", "screener_false_positive")),
    ]

    case = review_case(labels)

    assert case.id == "labels-9718091f"
    assert (case.question, case.cutoff) == (
        "How biased are LLMs used as judges?",
        "2026-10-07",
    )
    assert (case.gold, case.exclude) == (["2306.05685", "2410.00001"], ["2501.00002"])
