from bs4 import BeautifulSoup, Tag

from arxiv_agent.ingestion.models import ParsedPaper, Section

SECTION_LEVELS = {
    "ltx_section": 1,
    "ltx_appendix": 1,
    "ltx_subsection": 2,
    "ltx_subsubsection": 3,
    "ltx_paragraph": 4,
}


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text().split())


def _section_level(section: Tag) -> int | None:
    for css_class in section.get("class", []):
        if css_class in SECTION_LEVELS:
            return SECTION_LEVELS[css_class]
    return None


def _own_paragraphs(section: Tag) -> list[Tag]:
    return [
        p
        for p in section.select("div.ltx_para")
        if p.find_parent("section") is section
        and p.find_parent("div", class_="ltx_para") is None
    ]


def parse_arxiv_html(html: str) -> ParsedPaper:
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.select_one("h1.ltx_title_document")
    if title_tag is None:
        raise ValueError("not an arXiv HTML paper: no document title found")

    abstract_tag = soup.select_one("div.ltx_abstract")
    abstract = ""
    if abstract_tag is not None:
        abstract = "\n\n".join(_text(p) for p in abstract_tag.select("p.ltx_p"))

    sections = []
    for section in soup.find_all("section"):
        level = _section_level(section)
        if level is None or section.find_parent("div", class_="ltx_abstract"):
            continue
        heading = section.find(class_="ltx_title", recursive=False)
        sections.append(
            Section(
                title=_text(heading) if heading else "",
                level=level,
                text="\n\n".join(_text(p) for p in _own_paragraphs(section)),
            )
        )

    return ParsedPaper(title=_text(title_tag), abstract=abstract, sections=sections)
