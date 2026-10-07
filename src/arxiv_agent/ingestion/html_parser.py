import re

from bs4 import BeautifulSoup, Tag

from arxiv_agent.ingestion.models import ParsedPaper, Reference, Section, Table

SECTION_LEVELS = {
    "ltx_section": 1,
    "ltx_appendix": 1,
    "ltx_subsection": 2,
    "ltx_subsubsection": 3,
    "ltx_paragraph": 4,
}
ARXIV_ID_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
ARXIV_ID_IN_TEXT = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})", re.IGNORECASE)
WHITE_CHANNEL_MIN = 240
TINY_FONT_LIMITS = {"pt": 3.0, "px": 4.0, "%": 25.0, "em": 0.25}
HEX_COLOR = re.compile(r"^#([0-9a-f]{3}|[0-9a-f]{6})$")
FONT_SIZE = re.compile(r"^([\d.]+)\s*(pt|px|%|em)$")


def _text(tag: Tag, separator: str = "") -> str:
    return " ".join(tag.get_text(separator).split())


def _section_level(section: Tag) -> int | None:
    for css_class in section.get("class", []):
        if css_class in SECTION_LEVELS:
            return SECTION_LEVELS[css_class]
    return None


def _section_title(section: Tag | None) -> str:
    if section is None:
        return ""
    heading = section.find(class_="ltx_title", recursive=False)
    return _text(heading) if heading else ""


def _own_paragraphs(section: Tag) -> list[Tag]:
    return [
        p
        for p in section.select("div.ltx_para")
        if p.find_parent("section") is section
        and p.find_parent("div", class_="ltx_para") is None
    ]


def _style_declarations(style: str) -> dict[str, str]:
    declarations = {}
    for part in style.split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            declarations[key.strip().lower()] = value.strip().lower()
    return declarations


def _is_near_white(color: str) -> bool:
    if color == "white":
        return True
    match = HEX_COLOR.match(color)
    if match is None:
        return False
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(d * 2 for d in digits)
    return all(int(digits[i : i + 2], 16) >= WHITE_CHANNEL_MIN for i in (0, 2, 4))


def _is_tiny(font_size: str) -> bool:
    match = FONT_SIZE.match(font_size)
    return (
        match is not None and float(match.group(1)) < TINY_FONT_LIMITS[match.group(2)]
    )


def _is_hidden(tag: Tag) -> bool:
    declarations = _style_declarations(tag.get("style", ""))
    color = declarations.get("--ltx-fg-color") or declarations.get("color")
    font_size = declarations.get("font-size")
    return bool(
        (color and _is_near_white(color)) or (font_size and _is_tiny(font_size))
    )


def _remove_hidden_text(soup: BeautifulSoup) -> list[str]:
    hidden = []
    for tag in soup.select("[style]"):
        if tag.decomposed or tag.find_parent("svg") or not tag.get_text(strip=True):
            continue
        if _is_hidden(tag):
            hidden.append(_text(tag))
            tag.decompose()
    return hidden


def _replace_math_with_latex(soup: BeautifulSoup) -> None:
    for math in soup.find_all("math"):
        latex = math.get("alttext", "").replace("\\displaystyle", "").strip()
        math.replace_with(f"${latex}$" if latex else "")


def _parse_sections(soup: BeautifulSoup) -> list[Section]:
    sections = []
    for section in soup.find_all("section"):
        level = _section_level(section)
        if level is None or section.find_parent("div", class_="ltx_abstract"):
            continue
        sections.append(
            Section(
                title=_section_title(section),
                level=level,
                text="\n\n".join(_text(p) for p in _own_paragraphs(section)),
            )
        )
    return sections


def _table_markdown(figure: Tag) -> str:
    rows = [
        [_text(cell).replace("|", "\\|") for cell in tr.select(".ltx_td, .ltx_th")]
        for tr in figure.select(".ltx_tr")
    ]
    if not rows:
        return ""
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "|" + " --- |" * len(rows[0]))
    return "\n".join(lines)


def _parse_tables(soup: BeautifulSoup) -> list[Table]:
    tables = []
    for figure in soup.select("figure.ltx_table"):
        caption = figure.find("figcaption")
        tables.append(
            Table(
                section=_section_title(figure.find_parent("section")),
                caption=_text(caption) if caption else "",
                markdown=_table_markdown(figure),
            )
        )
    return tables


def _find_arxiv_id(item: Tag, text: str) -> str | None:
    for link in item.select("a[href]"):
        if match := ARXIV_ID_IN_URL.search(link["href"]):
            return match.group(1)
    if match := ARXIV_ID_IN_TEXT.search(text):
        return match.group(1)
    return None


def _parse_references(soup: BeautifulSoup) -> list[Reference]:
    references = []
    for item in soup.select("li.ltx_bibitem"):
        text = _text(item, separator=" ")
        references.append(
            Reference(
                ref_id=item.get("id", ""),
                text=text,
                arxiv_id=_find_arxiv_id(item, text),
            )
        )
    return references


def parse_arxiv_html(html: str) -> ParsedPaper:
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.select_one("h1.ltx_title_document")
    if title_tag is None:
        raise ValueError("not an arXiv HTML paper: no document title found")
    # \thanks footnotes (affiliations, emails) sit inside the title element.
    for note in title_tag.select(".ltx_pubnotes, .ltx_note"):
        note.decompose()

    hidden_text = _remove_hidden_text(soup)
    _replace_math_with_latex(soup)

    abstract_tag = soup.select_one("div.ltx_abstract")
    abstract = ""
    if abstract_tag is not None:
        abstract = "\n\n".join(_text(p) for p in abstract_tag.select("p.ltx_p"))

    return ParsedPaper(
        title=_text(title_tag),
        abstract=abstract,
        sections=_parse_sections(soup),
        tables=_parse_tables(soup),
        references=_parse_references(soup),
        hidden_text=hidden_text,
    )
