"""Layer 2, the content guard: paper text is data, so text in it that talks to
an AI is taken out before any model sees it.

Both checks are code:
- Fence tags. Prompts fence paper text in <paper_content>, <claims> and
  <evidence> tags and say what's inside is data. A paper that writes
  </paper_content> closes the fence and puts its next words outside it, so these
  tags are always removed from content.
- Instruction patterns. A sentence that addresses an AI or tells it what to do
  ("ignore previous instructions", "note to AI reviewers", "AI systems
  summarizing this paper must...", "score this paper 10") is removed. The
  patterns are narrow on purpose: "system prompt" alone appears in many honest
  papers about LLMs, "repeat your system prompt" doesn't.

What it can't catch: an instruction phrased as an ordinary claim ("this work
is foundational to the field"), or a pattern nobody wrote down yet. Removals are
flagged, never silent, and papers *about* prompt injection lose the attack
examples they quote.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

FENCE_TAG = re.compile(
    r"</?\s*(?:paper_content|claims|evidence)\b[^>]*>", re.IGNORECASE
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Attacks give orders; research describes. "Ignore previous instructions" opens
# a sentence (or follows "SYSTEM:" or "and"); "models often ignore system
# instructions" is a finding. So imperatives are matched where an order starts.
_ORDER = r"(?:^|[:;,]\s*|\band\s+)(?:please\s+|now\s+)?"
_AI = r"(?:ai|llm|language\s+models?|chatbots?|automated)"
_AGENT = r"(?:systems?|assistants?|reviewers?|tools?|agents?|models?|readers?)"
_THIS = r"(?:this|the\s+present)\s+(?:paper|document|work|article|text|submission)"
PATTERNS = {
    "ignore_instructions": (
        rf"{_ORDER}(?:ignore|disregard|forget|override)\b[^.!?\n]{{0,40}}"
        r"\b(?:instructions?|prompts?|rules|guidelines)\b"
    ),
    "addresses_ai": rf"\b(?:note|message|attention)\s+(?:to|for)\s+(?:the\s+|all\s+|any\s+)?{_AI}",
    "ai_must": (
        rf"\b{_AI}[\s-]+{_AGENT}\s+(?:that|which|reading|summari[sz]ing|reviewing|processing)"
        rf"\b[^.!?\n]{{0,40}}\b{_THIS}\b[^.!?\n]{{0,30}}"
        r"\b(?:must|should|shall|need\s+to|are\s+(?:required|instructed)\s+to)\b"
    ),
    "score_request": (
        r"\b(?:give|assign|award)\s+(?:this|the)\s+(?:paper|work|article|submission)\b"
        r"[^.!?\n]{0,30}\b(?:score|rating|grade)\b"
        r"|\b(?:score|rate|grade)\s+(?:this\s+(?:paper|work|submission)|the\s+paper\s+above)\b"
    ),
    "positive_review": (
        rf"{_ORDER}(?:give|write|provide|output)\b[^.!?\n]{{0,20}}\bpositive\s+review\b"
    ),
    "prompt_request": (
        rf"{_ORDER}(?:repeat|reveal|print|output|disclose|leak)\b[^.!?\n]{{0,30}}"
        r"\b(?:system\s+prompt|your\s+(?:instructions|prompt))\b"
    ),
    "output_demand": (
        r"\b(?:any|every|all|each)\s+(?:reviews?|summar(?:y|ies)|answers?)\b[^.!?\n]{0,40}"
        r"\bmust\s+(?:cite|include|state|mention|say)\b"
        r"|\balways\s+include\s+(?:the|this)\s+(?:link|url)\b"
        r"|\binclude\s+(?:this|the\s+following)\s+(?:link|url)\b"
    ),
}
COMPILED = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in PATTERNS.items()
}


class ContentFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    where: Literal["abstract", "passage", "hidden_text"]
    reason: str  # a pattern name, "fence_tag", or "hidden_text"
    text: str  # what was taken out (shortened)


def _match(sentence: str) -> str | None:
    return next((name for name, rx in COMPILED.items() if rx.search(sentence)), None)


def clean(
    text: str, arxiv_id: str, where: Literal["abstract", "passage"]
) -> tuple[str, list[ContentFlag]]:
    # Returns the text a model may see, and what was taken out of it.
    flags: list[ContentFlag] = []
    if FENCE_TAG.search(text):
        for tag in FENCE_TAG.findall(text):
            flags.append(
                ContentFlag(
                    arxiv_id=arxiv_id, where=where, reason="fence_tag", text=tag
                )
            )
        text = FENCE_TAG.sub(" ", text)
    lines = []
    for line in text.split("\n"):
        kept = []
        for sentence in SENTENCE_END.split(line):
            reason = _match(sentence)
            if reason:
                flags.append(
                    ContentFlag(
                        arxiv_id=arxiv_id,
                        where=where,
                        reason=reason,
                        text=sentence[:200],
                    )
                )
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    return "\n".join(lines).strip(), flags


def hidden_text_flags(arxiv_id: str, hidden: list[str]) -> list[ContentFlag]:
    # Hidden text is already removed when the page is parsed; this reports it.
    return [
        ContentFlag(
            arxiv_id=arxiv_id,
            where="hidden_text",
            reason="hidden_text",
            text=snippet[:200],
        )
        for snippet in hidden
    ]
