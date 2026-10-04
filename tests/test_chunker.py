from pathlib import Path

from arxiv_agent.ingestion.chunker import _approx_tokens, _section_paths
from arxiv_agent.ingestion.html_parser import parse_arxiv_html

FIXTURES = Path(__file__).parent / "fixtures"
PAPER = parse_arxiv_html(
    (FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")
)


def test_approx_tokens_round_up():
    assert _approx_tokens("") == 0
    assert _approx_tokens("abcd") == 1
    assert _approx_tokens("abcde") == 2


def test_section_paths_follow_nesting():
    assert _section_paths(PAPER.sections) == [
        ["1. Introduction"],
        ["2. Method"],
        ["2. Method", "2.1. Position Bias"],
        ["2. Method", "2.1. Position Bias", "2.1.1. Swapping"],
    ]
