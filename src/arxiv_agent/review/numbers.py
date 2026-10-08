"""Result numbers in text: the ones a claim stands or falls on.

Used by the Critic's code checks (a number in a review sentence must appear in
the passages it cites) and by the judge eval (which changes them on purpose).
The judge eval found the judge misses some changed numbers (98.2% -> 98.0%);
code doesn't.
"""

import re

# A number on its own: not inside a name (GPT-4, Qwen3-32B) or a formula;
# 12,000 counts as one number.
NUMBER = re.compile(r"(?<![\w.$\\-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w$-])")
# Numbers that aren't results: references to tables and sections, citations.
NOT_A_RESULT = re.compile(
    r"(?:table|figure|fig\.|section|sec\.|eq\.|equation|appendix|et al\.,?|\[)\s*$",
    re.IGNORECASE,
)


def result_numbers(text: str) -> list[re.Match]:
    # Decimals, percentages and numbers from 10 up, minus years and references.
    # Small counts ("3 epochs") are left out: often list items or footnotes.
    found = []
    for match in NUMBER.finditer(text):
        number = match.group(1)
        value = float(number.replace(",", ""))
        if 1900 <= value <= 2099 and "." not in number and "," not in number:
            continue  # a year
        if NOT_A_RESULT.search(text[: match.start()][-15:]):
            continue
        is_percent = text[match.end() : match.end() + 1] == "%"
        if not ("." in number or is_percent or value >= 10):
            continue
        found.append(match)
    return found


def _value(number: str) -> float:
    return float(number.replace(",", ""))


def missing_numbers(sentence: str, passages: list[str]) -> list[str]:
    # The sentence's result numbers that none of the passages contain,
    # compared by value: 62% matches 62.0%, 12000 matches 12,000.
    available = {
        _value(match.group(1))
        for passage in passages
        for match in NUMBER.finditer(passage)
    }
    return [
        match.group(1)
        for match in result_numbers(sentence)
        if _value(match.group(1)) not in available
    ]
