from pathlib import Path

import pytest

from arxiv_agent.ingestion.html_parser import parse_arxiv_html

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_HTML = (FIXTURES / "latexml_minimal.html").read_text(encoding="utf-8")
REAL_PAGE = Path(__file__).parent.parent / "data" / "html" / "2411.15594v6.html"


@pytest.fixture
def paper():
    return parse_arxiv_html(MINIMAL_HTML)


def test_title_and_abstract(paper):
    assert paper.title == "Judging the Judges: A Tiny Test Paper"
    assert paper.abstract == "We study how language models grade other models."


def test_abstract_section_is_not_a_section(paper):
    assert "Abstract" not in [s.title for s in paper.sections]


def test_sections_in_order_with_levels(paper):
    assert [(s.title, s.level) for s in paper.sections] == [
        ("1. Introduction", 1),
        ("2. Method", 1),
        ("2.1. Position Bias", 2),
        ("2.1.1. Swapping", 3),
    ]


def test_subsection_text_not_duplicated_in_parent(paper):
    method = paper.sections[1]

    assert method.text == "We compare judges against human raters."


def test_bullets_kept_once(paper):
    all_text = " ".join(s.text for s in paper.sections)

    assert all_text.count("A benchmark of judge agreement.") == 1


def test_bibliography_skipped(paper):
    assert all("References" not in s.title for s in paper.sections)
    assert all("Judging LLM-as-a-Judge" not in s.text for s in paper.sections)


def test_rejects_non_paper_html():
    with pytest.raises(ValueError):
        parse_arxiv_html("<html><body>hello</body></html>")


@pytest.mark.skipif(not REAL_PAGE.exists(), reason="real page not downloaded")
def test_real_page():
    paper = parse_arxiv_html(REAL_PAGE.read_text(encoding="utf-8"))

    assert paper.title == "A Survey on LLM-as-a-Judge"
    assert paper.abstract.startswith("Accurate and consistent evaluation")
    assert paper.sections[0].title == "1. Introduction"
    assert all(s.title for s in paper.sections)
    assert "Abstract" not in [s.title for s in paper.sections]
