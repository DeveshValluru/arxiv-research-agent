import asyncio
import json
import re
from pathlib import Path

from arxiv_agent.evals.redteam import (
    Injection,
    load_red_team,
    review_attack,
    review_library,
    screener_attack,
)
from arxiv_agent.guardrails.output import OutputGuard
from arxiv_agent.review.checkpoints import memory_checkpointer
from arxiv_agent.review.critic import Critic
from arxiv_agent.review.graph import REVIEW_PROMPTS, build_review_graph
from arxiv_agent.review.human import HumanReview
from arxiv_agent.review.nodes import ReviewNodes
from arxiv_agent.review.reader import Reader
from arxiv_agent.review.writer import Synthesizer
from tests.review_fakes import FakeChat

RED_TEAM = load_red_team(Path(__file__).parent.parent / "evals" / "redteam.json")
TARGET = RED_TEAM.off_topic[0]
DIRECT = next(i for i in RED_TEAM.screener_injections if i.id == "direct")
LINK = next(i for i in RED_TEAM.review_injections if i.id == "link")


def test_the_red_team_set_is_well_formed():
    on_topic = {p.arxiv_id for p in RED_TEAM.on_topic}
    assert RED_TEAM.review_target in on_topic
    assert not on_topic & {p.arxiv_id for p in RED_TEAM.off_topic}
    assert RED_TEAM.screener_injections[0].id == "clean"  # the control comes first
    for injection in RED_TEAM.review_injections:
        if injection.success:
            re.compile(injection.success)


def test_a_screener_attack_reports_the_targets_score():
    def screener(messages):
        # A fooled screener: anything mentioning a score of 10 gets one.
        fooled = "score of 10" in messages[1]["content"]
        return json.dumps(
            {
                "scores": [
                    {"arxiv_id": p.arxiv_id, "score": 9, "reason": "On topic."}
                    for p in RED_TEAM.on_topic
                ]
                + [
                    {
                        "arxiv_id": TARGET.arxiv_id,
                        "score": 10 if fooled else 1,
                        "reason": "r",
                    }
                ]
            }
        )

    def attack(injection, guard_content):
        chat = FakeChat({"screener-llm": screener})
        nodes = ReviewNodes(chat, toolbox=None, guard_content=guard_content)
        return asyncio.run(screener_attack(nodes, RED_TEAM, TARGET, injection))

    clean = attack(RED_TEAM.screener_injections[0], guard_content=False)
    unguarded = attack(DIRECT, guard_content=False)
    guarded = attack(DIRECT, guard_content=True)

    assert (clean.score, clean.kept) == (1, False)
    assert (unguarded.score, unguarded.kept) == (10, True)  # the harness sees it work
    assert (guarded.score, guarded.kept) == (1, False)  # the content guard stops it


def review(writer: FakeChat, injection: Injection, guard_content: bool = False):
    # The content guard is off unless a test turns it on: these tests check
    # the layers after it.
    judge = FakeChat({"critic-llm": '{"verdict": "supported", "reason": "r"}'})
    graph = build_review_graph(
        ReviewNodes(writer, toolbox=None),
        Reader(
            writer, review_library(RED_TEAM, injection), guard_content=guard_content
        ),
        Synthesizer(writer),
        Critic(judge),
        HumanReview(toolbox=None),
        checkpointer=memory_checkpointer(),
    )
    return asyncio.run(
        review_attack(graph, RED_TEAM, injection, OutputGuard(REVIEW_PROMPTS), "t1")
    )


def reader_that_quotes(messages):
    # Quotes the first sentence of P1 (and, for the target, the injection).
    passage = messages[1]["content"].split("[P1] (Results)\n")[1].split("\n")[0]
    first = passage.split(". ")[0] + "."
    claims = [{"claim": first, "passage": "P1", "quote": first}]
    if "evil.example.com" in passage:
        claims.append({"claim": "See the link.", "passage": "P1", "quote": LINK.text})
    return json.dumps({"claims": claims})


def test_a_link_the_synthesizer_was_fooled_into_writing_is_removed_by_the_critic():
    # The link reaches the claims and every draft; the Critic's code check
    # (no ids or links written by the model) rejects it, and after the last
    # rewrite the sentence is removed. The output guard never sees it.
    writer = FakeChat(
        {
            "reader-llm": reader_that_quotes,
            "synthesizer-llm": (
                "Judges prefer the first answer [K1]. "
                "Details are at https://evil.example.com/llm-judges [K5]."
            ),
        }
    )

    result = review(writer, LINK)

    assert (result.in_claims, result.in_draft, result.shipped) == (True, True, False)
    assert (result.removed, result.blocked, result.drafts) == (1, 0, 3)


def test_a_prompt_leak_that_passes_the_critic_is_blocked_by_the_output_guard():
    # The Critic judges support, not leaks: this sentence cites a real claim
    # and passes. Only the output guard catches it.
    leak = next(i for i in RED_TEAM.review_injections if i.id == "prompt_leak")
    writer = FakeChat(
        {
            "reader-llm": reader_that_quotes,
            "synthesizer-llm": (
                "Judges prefer the first answer [K1]. Use only the claims and do not "
                "add facts from your own knowledge [K2]."
            ),
        }
    )

    result = review(writer, leak)

    assert (result.in_draft, result.shipped) == (True, False)
    assert (result.removed, result.blocked) == (0, 1)


def test_an_attack_that_never_reaches_the_claims_shows_up_nowhere():
    writer = FakeChat(
        {
            "reader-llm": '{"claims": [{"claim": "c", "passage": "P1", "quote": "We evaluate six judge models on 4,000 pairwise comparisons drawn from three benchmarks."}]}',
            "synthesizer-llm": "Judges were evaluated at scale [K1].",
        }
    )

    result = review(writer, LINK)

    assert (result.in_claims, result.in_draft, result.shipped, result.blocked) == (
        False,
        False,
        False,
        0,
    )


def test_with_the_content_guard_the_injection_never_reaches_the_reader():
    writer = FakeChat(
        {
            "reader-llm": reader_that_quotes,
            "synthesizer-llm": "Judges prefer the first answer [K1].",
        }
    )

    result = review(writer, LINK, guard_content=True)

    assert (result.in_claims, result.in_draft, result.shipped) == (False, False, False)
    prompts = " ".join(m[1]["content"] for m in writer.prompts("reader-llm"))
    assert "evil.example.com" not in prompts
