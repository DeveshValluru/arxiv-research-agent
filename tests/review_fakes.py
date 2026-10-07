"""Fakes for the review graph: scripted models, a library without a database,
and fake arXiv and OpenAlex clients behind the real MCP servers.
"""

import asyncio
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime

from arxiv_agent.clients.arxiv import PaperSummary
from arxiv_agent.clients.openalex import Work
from arxiv_agent.library import Passage
from arxiv_agent.llm import Completion, Usage
from arxiv_agent.mcp_servers.arxiv_server import create_server as arxiv_server
from arxiv_agent.mcp_servers.openalex_server import create_server as openalex_server
from arxiv_agent.review.checkpoints import memory_checkpointer
from arxiv_agent.review.critic import Critic
from arxiv_agent.review.graph import ReviewRun, build_review_graph, run_review
from arxiv_agent.review.human import HumanReview
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.reader import Reader
from arxiv_agent.review.state import Budget
from arxiv_agent.review.writer import Synthesizer
from arxiv_agent.tools.toolbox import McpToolbox

QUESTION = "How biased are LLM judges?"


def paper(
    arxiv_id: str, title: str, published: datetime = datetime(2024, 5, 1, tzinfo=UTC)
) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        version=1,
        title=title,
        authors=["Ada Lovelace", "Alan Turing", "Grace Hopper"],
        abstract=f"An abstract about {title.lower()}.",
        published=published,
        updated=published,
        primary_category="cs.CL",
        categories=["cs.CL"],
    )


POSITION = paper("2406.07791", "Judging the Judges: Position Bias")
SWAP = paper("2305.17926", "Large Language Models are not Fair Evaluators")
AGREEMENT = paper("2306.05685", "Judging LLM-as-a-Judge with MT-Bench")
PREJUDICE = paper("2410.02736", "Justice or Prejudice? Biases in LLM-as-a-Judge")
JUDGELM = paper("2310.17631", "JudgeLM: Fine-tuned Judges")
FOLLOWUP = paper(
    "2601.00001", "A Follow-up on Judge Bias", datetime(2026, 1, 5, tzinfo=UTC)
)
PAPERS = {
    p.arxiv_id: p for p in (POSITION, SWAP, AGREEMENT, PREJUDICE, JUDGELM, FOLLOWUP)
}

# One finding per paper: the passage the library returns contains it, and the
# scripted Reader quotes it.
FINDINGS = {
    POSITION.arxiv_id: "LLM judges prefer the answer that is shown first in pairwise comparisons.",
    SWAP.arxiv_id: "Swapping the order of two answers changes the verdict of GPT-4 in many cases.",
    AGREEMENT.arxiv_id: "GPT-4 as a judge agrees with human experts over 80% of the time.",
    PREJUDICE.arxiv_id: "We identify twelve kinds of bias that affect LLM judges.",
    JUDGELM.arxiv_id: "Fine-tuned judge models reach high agreement with GPT-4 judgments.",
    FOLLOWUP.arxiv_id: "Calibrating judge scores reduces position bias in our experiments.",
}

PLAN = json.dumps(
    {
        "sub_queries": ["position bias LLM judges", "LLM judge human agreement"],
        "criteria": ["evaluates LLMs used as judges", "reports a bias or agreement"],
    }
)
SCORES = json.dumps(
    {
        "scores": [
            {"arxiv_id": "2406.07791", "score": 9, "reason": "Measures position bias."},
            {
                "arxiv_id": "2305.17926",
                "score": 7,
                "reason": "Shows judges are unfair.",
            },
            {"arxiv_id": "2306.05685", "score": 4, "reason": "Mostly a benchmark."},
            {"arxiv_id": "2410.02736", "score": 10, "reason": "Catalogues biases."},
            {"arxiv_id": "2310.17631", "score": 5, "reason": "About training judges."},
            {"arxiv_id": "2601.00001", "score": 6, "reason": "Reduces a bias."},
            {"arxiv_id": "9999.99999", "score": 10, "reason": "Invented by the model."},
        ]
    }
)
VERSION_SUFFIX = re.compile(r"v\d+$")
TITLE = re.compile(r"Title: (.+)")


def bibliography_html(cited: list[str]) -> str:
    items = "".join(
        f'<li class="ltx_bibitem" id="bib.bib{i}">Some paper. '
        f"arXiv preprint arXiv:{arxiv_id}.</li>"
        for i, arxiv_id in enumerate(cited)
    )
    return (
        '<html><body><h1 class="ltx_title ltx_title_document">T</h1>'
        f'<section class="ltx_bibliography"><ul>{items}</ul></section></body></html>'
    )


class FakeArxiv:
    # The real arXiv server builds "all:position AND all:bias"-style queries;
    # this fake answers by keyword. bibliographies: arXiv id -> cited ids.
    def __init__(self, bibliographies: dict[str, list[str]] | None = None) -> None:
        self.bibliographies = bibliographies or {}
        self.queries: list[str] = []

    def search_papers(self, query, max_results=5):
        self.queries.append(query)
        if "position" in query:
            return [POSITION, SWAP][:max_results]
        if "agreement" in query:
            return [SWAP, AGREEMENT][:max_results]
        return []

    def get_metadata(self, ids):
        bare = [VERSION_SUFFIX.sub("", arxiv_id) for arxiv_id in ids]
        return {arxiv_id: PAPERS[arxiv_id] for arxiv_id in bare if arxiv_id in PAPERS}

    def fetch_html(self, arxiv_id, version=None):
        cited = self.bibliographies.get(arxiv_id)
        return None if cited is None else bibliography_html(cited)


class FakeOpenAlex:
    # citing: arXiv id -> the arXiv ids of papers citing it.
    def __init__(self, citing: dict[str, list[str]] | None = None) -> None:
        self.citing = citing or {}

    @staticmethod
    def _work(arxiv_id: str) -> Work:
        return Work(
            openalex_id=f"W{arxiv_id}",
            title=PAPERS[arxiv_id].title,
            year=2024,
            cited_by_count=10,
            venue=None,
            doi=None,
            arxiv_id=arxiv_id,
            authors=[],
            author_count=0,
        )

    def get_work_by_arxiv_id(self, arxiv_id):
        return self._work(arxiv_id) if arxiv_id in PAPERS else None

    def get_citations(self, openalex_id, limit=10, sort="most_cited"):
        citing = self.citing.get(openalex_id.removeprefix("W"), [])
        return len(citing), [self._work(arxiv_id) for arxiv_id in citing][:limit]


class FakeLibrary:
    # sources: arXiv id -> what ensure_ingested reports (default "full_text").
    # on_ingest: called with each arXiv id as it's ingested (to move a fake
    # clock, or to watch what happens meanwhile).
    def __init__(
        self,
        sources: dict[str, str] | None = None,
        on_ingest: Callable[[str], None] | None = None,
    ) -> None:
        self.sources = sources or {}
        self.on_ingest = on_ingest
        self.ingested: list[str] = []

    def ensure_ingested(self, arxiv_id, version):
        if self.on_ingest:
            self.on_ingest(arxiv_id)
        self.ingested.append(arxiv_id)
        return self.sources.get(arxiv_id, "full_text")

    def passages(self, arxiv_id, version, question, k):
        return [
            Passage(
                chunk_id=f"{arxiv_id}v{version}:0003",
                section="4 Results",
                text=f"We ran the study. {FINDINGS[arxiv_id]} Details follow.",
            )
        ][:k]


Reply = str | list[str] | Callable[[list[dict]], str]
CALL_USAGE = Usage(llm_calls=1, prompt_tokens=100, completion_tokens=10)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class FakeChat:
    # Replies by call name: a fixed string, a list used in order (the last one
    # repeats), or a function of the messages. Records every call; each one
    # "uses" CALL_USAGE.
    def __init__(self, replies: dict[str, Reply]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, list[dict]]] = []

    async def complete(self, messages, *, name, max_tokens=800, temperature=0.0):
        self.calls.append((name, messages))
        reply = self.replies[name]
        if callable(reply):
            text = reply(messages)
        elif isinstance(reply, list):
            text = reply[min(len(self.prompts(name)), len(reply)) - 1]
        else:
            text = reply
        return Completion(text=text, usage=CALL_USAGE)

    def prompts(self, name: str) -> list[list[dict]]:
        return [messages for call_name, messages in self.calls if call_name == name]


def faithful_reader(messages: list[dict]) -> str:
    # Quotes the paper's finding word for word, from passage P1.
    title = TITLE.search(messages[1]["content"]).group(1)
    arxiv_id = next(i for i, p in PAPERS.items() if p.title == title)
    finding = FINDINGS[arxiv_id]
    return json.dumps(
        {
            "claims": [
                {"claim": f"Paraphrase: {finding}", "passage": "P1", "quote": finding}
            ]
        }
    )


def strict_judge(messages: list[dict]) -> str:
    # Calls a sentence overstated when it says "all"; otherwise supported.
    sentence = messages[1]["content"].split("\n")[0]
    if " all " in f" {sentence.lower()} ":
        return '{"verdict": "overstated", "reason": "The paper tested one model."}'
    return '{"verdict": "supported", "reason": "The passage says so."}'


# With the default fakes, two papers are kept: K1 is POSITION, K2 is SWAP.
GOOD_DRAFT = (
    "LLM judges favour the answer shown first [K1]. "
    "Swapping the answer order can flip GPT-4's verdict [K2]."
)


def chat(**overrides: Reply) -> FakeChat:
    return FakeChat(
        {
            "planner-llm": PLAN,
            "screener-llm": SCORES,
            "reader-llm": faithful_reader,
            "synthesizer-llm": GOOD_DRAFT,
        }
        | overrides
    )


def judge(reply: Reply = strict_judge) -> FakeChat:
    return FakeChat({"critic-llm": reply})


class Harness:
    # A review graph wired to fakes. Every run() opens fresh MCP connections and
    # builds a fresh graph, like a new process would, but they all share one
    # checkpointer: so a test can pause, "restart", and resume the same review.
    def __init__(
        self,
        writer: FakeChat,
        judge_model: FakeChat | None = None,
        arxiv: FakeArxiv | None = None,
        openalex: FakeOpenAlex | None = None,
        library: FakeLibrary | None = None,
        max_revisions: int = 2,
        clock: Callable[[], float] = time.time,
        checkpointer=None,
        **finder_settings,
    ) -> None:
        self.writer = writer
        self.judge = judge_model or judge()
        self.arxiv = arxiv or FakeArxiv()
        self.openalex = openalex or FakeOpenAlex()
        self.library = library or FakeLibrary()
        self.max_revisions = max_revisions
        self.clock = clock
        self.checkpointer = checkpointer or memory_checkpointer()
        self.finder_settings = finder_settings

    def run(self, thread_id: str = "review-1", **review_settings) -> ReviewRun:
        async def go():
            servers = {
                "arxiv": arxiv_server(self.arxiv),
                "openalex": openalex_server(self.openalex),
            }
            async with McpToolbox(servers) as toolbox:
                graph = build_review_graph(
                    ReviewNodes(self.writer, toolbox, **self.finder_settings),
                    Reader(self.writer, self.library, clock=self.clock),
                    Synthesizer(self.writer),
                    Critic(self.judge, max_revisions=self.max_revisions),
                    HumanReview(toolbox, clock=self.clock),
                    clock=self.clock,
                    checkpointer=self.checkpointer,
                )
                return await run_review(
                    QUESTION, graph, thread_id=thread_id, **review_settings
                )

        return asyncio.run(go())


def run(
    writer: FakeChat,
    judge_model: FakeChat | None = None,
    arxiv: FakeArxiv | None = None,
    openalex: FakeOpenAlex | None = None,
    library: FakeLibrary | None = None,
    max_revisions: int = 2,
    budget: Budget | None = None,
    clock: Callable[[], float] = time.time,
    on_event: Callable[[str, dict], None] | None = None,
    **finder_settings,
):
    # One review start to finish (no pause): returns (state, trace_id).
    harness = Harness(
        writer,
        judge_model,
        arxiv,
        openalex,
        library,
        max_revisions,
        clock,
        **finder_settings,
    )
    review = harness.run(budget=budget, on_event=on_event)
    return review.state, review.trace_id
