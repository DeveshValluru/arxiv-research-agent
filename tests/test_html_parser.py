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


def test_math_becomes_latex(paper):
    swapping = paper.sections[3]

    assert swapping.text == "Swapping the order reduces the bias by $\\Delta b$."


def test_table_as_markdown(paper):
    [table] = paper.tables

    assert table.section == "2. Method"
    assert table.caption == "Table 1. Judge agreement with humans."
    assert table.markdown == (
        "| Judge | Agreement |\n| --- | --- |\n| GPT-4 | 85% |\n| Llama-3 | 78% |"
    )


def test_table_text_not_in_section_text(paper):
    assert all("Llama-3" not in s.text for s in paper.sections)


def test_references_with_arxiv_ids(paper):
    assert [(r.ref_id, r.arxiv_id) for r in paper.references] == [
        ("bib.bib1", "2306.05685"),
        ("bib.bib2", "2305.18248"),
        ("bib.bib3", None),
    ]
    assert paper.references[0].text.startswith("Zheng et al. (2023) Lianmin Zheng")


@pytest.mark.skipif(not REAL_PAGE.exists(), reason="real page not downloaded")
def test_real_page():
    paper = parse_arxiv_html(REAL_PAGE.read_text(encoding="utf-8"))
    all_text = paper.abstract + " ".join(s.text for s in paper.sections)

    assert paper.title == "A Survey on LLM-as-a-Judge"
    assert paper.abstract.startswith("Accurate and consistent evaluation")
    assert paper.sections[0].title == "1. Introduction"
    assert all(s.title for s in paper.sections)
    assert "Abstract" not in [s.title for s in paper.sections]
    assert len(paper.tables) == 4
    assert len(paper.references) == 226
    assert sum(r.arxiv_id is not None for r in paper.references) >= 114
    assert "\\displaystyle" not in all_text
    assert "\u200b" not in all_text
