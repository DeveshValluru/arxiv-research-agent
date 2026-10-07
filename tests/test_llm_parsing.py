import pytest

from arxiv_agent.llm import LLMOutputError, parse_json_object, strip_thinking
from arxiv_agent.review.state import Plan


def test_strip_thinking():
    assert strip_thinking("<think>hmm {not json}</think>\n\nAnswer.") == "Answer."


def test_parses_json_wrapped_in_a_fence_and_a_thinking_block():
    raw = (
        "<think>first {draft}</think>Here you go:\n```json\n"
        '{"sub_queries": ["llm judge bias", "position bias"], "criteria": ["about judges"]}'
        "\n```"
    )
    plan = parse_json_object(raw, Plan)
    assert plan.sub_queries == ["llm judge bias", "position bias"]


def test_missing_json_is_an_error():
    with pytest.raises(LLMOutputError, match="no JSON object"):
        parse_json_object("I would search for judges.", Plan)


def test_json_that_breaks_the_schema_is_an_error():
    too_many = '{"sub_queries": ["a", "b", "c", "d", "e", "f", "g"], "criteria": ["x"]}'
    with pytest.raises(LLMOutputError, match="doesn't fit Plan"):
        parse_json_object(too_many, Plan)
