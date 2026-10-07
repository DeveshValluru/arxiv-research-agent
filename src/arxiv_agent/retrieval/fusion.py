RRF_K = 60  # from the original RRF paper (Cormack et al., 2009); rarely needs tuning


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    # Combine rankings by position only, so methods whose scores live on
    # different scales (cosine vs BM25) can be merged. An item ranked by
    # several lists collects several shares and rises above items that only
    # one list liked. Ties keep the order items were first seen in.
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])
