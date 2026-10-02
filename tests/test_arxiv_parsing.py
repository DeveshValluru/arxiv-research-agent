from datetime import datetime
from pathlib import Path

import pytest

from arxiv_agent.clients.arxiv import _clean, _parse_feed, _split_id

FIXTURES = Path(__file__).parent / "fixtures"

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: no results</title>
</feed>"""


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("http://arxiv.org/abs/2411.16594v7", ("2411.16594", 7)),
        ("http://arxiv.org/abs/2410.11594v1", ("2410.11594", 1)),
        ("http://arxiv.org/abs/hep-th/9901001v1", ("hep-th/9901001", 1)),
        ("http://arxiv.org/abs/2301.00001v12", ("2301.00001", 12)),
    ],
)
def test_split_id(raw_id, expected):
    assert _split_id(raw_id) == expected


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-an-id",
        "http://arxiv.org/abs/2411.16594",
        "http://arxiv.org/api/errors#bad",
    ],
)
def test_split_id_rejects_bad_ids(bad_id):
    with pytest.raises(ValueError):
        _split_id(bad_id)


def test_clean_collapses_all_whitespace():
    assert (
        _clean("  LLM Judges\n  with Human\tRaters  ") == "LLM Judges with Human Raters"
    )


def test_parse_feed_cleans_line_breaks_and_keeps_latex():
    xml = (FIXTURES / "arxiv_search_SYNTHETIC.xml").read_text(encoding="utf-8")

    papers = _parse_feed(xml)

    assert len(papers) == 3
    assert "\n" not in papers[0].title
    assert papers[2].title.endswith("($\\alpha$-Calibrated Scoring)")
    assert papers[2].categories == ["cs.LG", "cs.CL", "stat.ML"]
    assert isinstance(papers[0].published, datetime)


def test_parse_feed_with_no_results_returns_empty_list():
    assert _parse_feed(EMPTY_FEED) == []
