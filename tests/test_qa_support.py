from arxiv_agent.qa.answerer import Answerer
from arxiv_agent.qa.checker import REFUSAL, check_answer
from arxiv_agent.qa.support import SentenceSupport, SupportChecker, apply_support
from tests.test_answerer import (
    HITS,
    FakeHTTPError,
    FakeLangfuse,
    FakeLLM,
    FakeRetriever,
)

# HITS: [S1] "Swapping the order reduces the bias by $\Delta b$."
#       [S2] "Judges prefer the first answer shown. ..."
TRUE = "Swapping the order reduces the bias [S1]."
OVERSTATED = "All judges always prefer the first answer [S2]."
FIXED = "Judges often prefer the first answer shown [S2]."
FIRST = f"{TRUE} {OVERSTATED}"  # the answer before the check


def verdict_by_sentence(messages):
    # A fake judge: overstated when the sentence says "always", else supported.
    sentence = messages[1]["content"].split("\n")[0]
    if "always" in sentence:
        return '{"verdict": "overstated", "reason": "The source tested one setting."}'
    return '{"verdict": "supported", "reason": "The source says so."}'


class FakeJudge:
    # Called from the checker's threads, so each reply stays local to its call.
    def __init__(self, reply=verdict_by_sentence) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def chat_completion(self, messages, model=None, max_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        reply = self.reply(messages) if callable(self.reply) else self.reply
        return FakeLLM(reply).chat_completion(messages, model, max_tokens)


def checker(judge: FakeJudge) -> SupportChecker:
    return SupportChecker(
        model="test/judge",
        providers=["j"],
        clients={"j": judge},
        langfuse=FakeLangfuse(),
    )


def verdicts(support: list[SentenceSupport]) -> list[str]:
    return [s.verdict for s in support]


def test_each_cited_sentence_is_judged_against_the_sources_it_cites():
    judge = FakeJudge()

    support = checker(judge).check(f"{TRUE} {OVERSTATED}", HITS)

    assert verdicts(support) == ["supported", "overstated"]
    assert [s.sources for s in support] == [[1], [2]]
    evidence = [call["messages"][1]["content"] for call in judge.calls]
    assert any("[S2] (2. Method > 2.1. Position Bias)" in e for e in evidence)


def test_a_wrong_number_is_caught_in_code_without_asking_the_judge():
    judge = FakeJudge()

    [support] = checker(judge).check("The bias drops by 42% [S1].", HITS)

    assert (support.verdict, support.reason) == (
        "unsupported",
        "42 isn't in the sources it cites",
    )
    assert judge.calls == []


def test_uncited_sentences_are_kept_and_a_failing_judge_doesnt_count_against_them():
    support = checker(FakeJudge(reply="no idea")).check(
        f"The paper doesn't cover costs. {TRUE}", HITS
    )
    assert verdicts(support) == ["uncited", "unchecked"]


def test_failing_sentences_are_removed_and_citations_recounted():
    answer = check_answer(f"{TRUE} {OVERSTATED}", n_sources=2)
    support = checker(FakeJudge()).check(answer.text, HITS)

    checked = apply_support(answer, support)

    assert (checked.text, checked.status, checked.cited) == (TRUE, "answered", [1])
    assert checked.problems == [
        f"removed (overstated: The source tested one setting.): {OVERSTATED}"
    ]


def test_nothing_supported_left_means_the_answer_refuses():
    answer = check_answer(f"The paper doesn't cover costs. {OVERSTATED}", n_sources=2)
    support = checker(FakeJudge()).check(answer.text, HITS)

    checked = apply_support(answer, support)

    assert (checked.text, checked.status) == (REFUSAL, "refused")


def answerer(
    llm: FakeLLM, judge: FakeJudge | None = None, langfuse=None, repair=True
) -> Answerer:
    return Answerer(
        FakeRetriever(HITS),
        model="test/fake-llm",
        providers=["a"],
        clients={"a": llm},
        langfuse=langfuse or FakeLangfuse(),
        sleep=lambda seconds: None,
        support=checker(judge or FakeJudge()),
        repair=repair,
    )


def test_the_answerer_runs_the_check_and_records_it():
    langfuse = FakeLangfuse()

    result = answerer(FakeLLM(FIRST), langfuse=langfuse, repair=False).ask(
        "Why swap?", "2499.00001", 1
    )

    assert (result.answer.text, result.repair) == (TRUE, None)
    assert verdicts(result.support) == ["supported", "overstated"]
    assert result.support_ms > 0
    [span] = langfuse.named("support-check")
    assert span.updates["level"] == "WARNING"


def test_a_refusal_is_not_checked():
    judge = FakeJudge()

    result = answerer(FakeLLM(REFUSAL), judge).ask("What about cost?", "2499.00001", 1)

    assert (result.answer.status, result.support, judge.calls) == ("refused", [], [])


def test_a_failing_sentence_is_rewritten_once_and_only_the_rewrite_is_judged_again():
    judge, llm = FakeJudge(), FakeLLM(FIRST, f"{TRUE} {FIXED}")

    result = answerer(llm, judge).ask("Why swap?", "2499.00001", 1)

    assert (result.answer.text, result.answer.cited) == (f"{TRUE} {FIXED}", [1, 2])
    assert result.repair.outcome == "repaired"
    assert verdicts(result.repair.support) == ["supported", "supported"]
    assert result.answer.problems == [
        f"repaired (overstated: The source tested one setting.): {OVERSTATED}"
    ]
    # The repair turn: the first answer, then which sentence failed and why.
    *_, said, asked = llm.calls[1]["messages"]
    assert said == {"role": "assistant", "content": FIRST}
    assert OVERSTATED in asked["content"]
    assert "The source tested one setting." in asked["content"]
    assert len(judge.calls) == 3  # 2 sentences, then only the new one


def test_what_still_fails_after_the_rewrite_is_removed():
    # Rewritten word for word: the known verdict stands, no new judge call.
    judge = FakeJudge()

    result = answerer(FakeLLM(FIRST), judge).ask("Why swap?", "2499.00001", 1)

    assert (result.answer.text, result.repair.outcome) == (TRUE, "repaired")
    assert len(judge.calls) == 2


def test_a_wrong_number_added_by_the_rewrite_is_caught():
    rewrite = f"{TRUE} Judges prefer the first answer in 80% of cases [S2]."

    result = answerer(FakeLLM(FIRST, rewrite)).ask("Why swap?", "2499.00001", 1)

    assert result.answer.text == TRUE
    assert result.repair.support[1].reason == "80 isn't in the sources it cites"


def test_an_unusable_rewrite_falls_back_to_removing_the_failing_sentences():
    result = answerer(FakeLLM(FIRST, REFUSAL)).ask("Why swap?", "2499.00001", 1)

    assert result.answer.text == TRUE  # not the refusal
    assert (result.repair.outcome, result.repair.reason) == (
        "fallback",
        "rewrite unusable: refused",
    )


def test_no_provider_for_the_rewrite_falls_back_too():
    llm = FakeLLM(FIRST, FakeHTTPError(503))

    result = answerer(llm).ask("Why swap?", "2499.00001", 1)

    assert (result.answer.text, result.repair.outcome) == (TRUE, "fallback")
    assert result.repair.reason.startswith("no provider")
