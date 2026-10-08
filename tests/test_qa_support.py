from arxiv_agent.qa.answerer import Answerer
from arxiv_agent.qa.checker import REFUSAL, check_answer
from arxiv_agent.qa.support import SentenceSupport, SupportChecker, apply_support
from tests.test_answerer import HITS, FakeLangfuse, FakeLLM, FakeRetriever

# HITS: [S1] "Swapping the order reduces the bias by $\Delta b$."
#       [S2] "Judges prefer the first answer shown. ..."
TRUE = "Swapping the order reduces the bias [S1]."
OVERSTATED = "All judges always prefer the first answer [S2]."


def verdict_by_sentence(messages):
    # A fake judge: overstated when the sentence says "always", else supported.
    sentence = messages[1]["content"].split("\n")[0]
    if "always" in sentence:
        return '{"verdict": "overstated", "reason": "The source tested one setting."}'
    return '{"verdict": "supported", "reason": "The source says so."}'


class FakeJudge(FakeLLM):
    def __init__(self, reply=verdict_by_sentence) -> None:
        super().__init__("")
        self.reply = reply

    def chat_completion(self, messages, model=None, max_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        self.outcomes = [self.reply(messages) if callable(self.reply) else self.reply]
        return super().chat_completion(messages, model, max_tokens)


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


def test_the_answerer_runs_the_check_and_records_it():
    langfuse = FakeLangfuse()
    answerer = Answerer(
        FakeRetriever(HITS),
        model="test/fake-llm",
        providers=["a"],
        clients={"a": FakeLLM(f"{TRUE} {OVERSTATED}")},
        langfuse=langfuse,
        support=checker(FakeJudge()),
    )

    result = answerer.ask("Why swap?", "2499.00001", 1)

    assert result.answer.text == TRUE
    assert verdicts(result.support) == ["supported", "overstated"]
    assert result.support_ms > 0
    [span] = langfuse.named("support-check")
    assert span.updates["level"] == "WARNING"


def test_a_refusal_is_not_checked():
    judge = FakeJudge()
    answerer = Answerer(
        FakeRetriever(HITS),
        model="test/fake-llm",
        providers=["a"],
        clients={"a": FakeLLM(REFUSAL)},
        langfuse=FakeLangfuse(),
        support=checker(judge),
    )

    result = answerer.ask("What about cost?", "2499.00001", 1)

    assert (result.answer.status, result.support, judge.calls) == ("refused", [], [])
