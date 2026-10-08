"""Build the literature-review eval set from survey papers' bibliographies.

    uv run --env-file .env python scripts/build_review_eval.py

For each survey below: read its bibliography from its arXiv HTML page, keep
the references with an arXiv id, look them up, and keep those submitted before
the survey's first version (the date the reviews will run "as of"). The
question is written by hand to match the survey's scope, phrased the way a
person would ask. The result goes to evals/review_surveys.jsonl.

The two LLM-as-a-judge surveys share a question on purpose: the same review is
scored against two expert bibliographies, which shows how much the gold set
itself moves the score.
"""

import logging
from pathlib import Path

from arxiv_agent.clients.arxiv import ArxivClient, shared_arxiv_client
from arxiv_agent.evals.review_eval import SurveyCase
from arxiv_agent.ingestion.html_cache import load_html
from arxiv_agent.ingestion.html_parser import parse_arxiv_html

OUTPUT = Path("evals/review_surveys.jsonl")
JUDGE_QUESTION = (
    "How are large language models used as judges to evaluate other models, "
    "and how reliable and biased are they?"
)
RAG_QUESTION = (
    "How does retrieval-augmented generation improve large language models, "
    "and what retrieval and augmentation methods does it use?"
)
HALLUCINATION_QUESTION = (
    "What causes hallucinations in large language models, and how can they be "
    "detected and reduced?"
)
TOOLS_QUESTION = (
    "How do large language models learn to use external tools, and how is "
    "their tool use evaluated?"
)
SURVEYS = [
    ("llm-judge-a", "2411.15594", JUDGE_QUESTION),
    ("llm-judge-b", "2412.05579", JUDGE_QUESTION),
    ("rag", "2312.10997", RAG_QUESTION),
    ("hallucination", "2311.05232", HALLUCINATION_QUESTION),
    ("tool-learning", "2405.17935", TOOLS_QUESTION),
]


def build_case(
    client: ArxivClient, case_id: str, survey_id: str, question: str
) -> SurveyCase:
    survey = client.get_metadata([survey_id])[survey_id]
    html = load_html(client, survey_id, survey.version)  # cached after the first run
    if html is None:
        raise SystemExit(f"{survey_id} has no HTML version, so no bibliography")
    references = parse_arxiv_html(html).references
    cited = list(dict.fromkeys(r.arxiv_id for r in references if r.arxiv_id))
    found = client.get_metadata(cited)
    cutoff = survey.published.date().isoformat()
    gold = sorted(
        arxiv_id
        for arxiv_id in cited
        if arxiv_id in found and found[arxiv_id].published.date().isoformat() < cutoff
    )
    return SurveyCase(
        id=case_id,
        survey=survey_id,
        version=survey.version,
        title=survey.title,
        cutoff=cutoff,
        question=question,
        references=len(references),
        with_arxiv_id=len(cited),
        gold=gold,
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    client = shared_arxiv_client()
    lines = []
    for case_id, survey_id, question in SURVEYS:
        case = build_case(client, case_id, survey_id, question)
        lines.append(case.model_dump_json())
        print(
            f"{case.id:<14} {case.survey}v{case.version}  as of {case.cutoff}  "
            f"{case.references} refs, {case.with_arxiv_id} with an arXiv id, "
            f"{len(case.gold)} gold  | {case.title[:50]}"
        )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
