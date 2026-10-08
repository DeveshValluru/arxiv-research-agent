"""Layer 4, the output guard: the last checks on anything the system ships.

All code, no model: free, instant, and never wrong in a new way twice.
- links: every link points at an allowed host. A model writing links invents
  them (seen in Phase 4: it built arXiv URLs from ids), and a link is the
  easiest way to send a reader somewhere harmful.
- prompt leaks: no run of LEAK_WORDS consecutive words from one of our system
  prompts. A leaked prompt shows a reader (or an attacker steering the model
  through a paper) exactly what the rules are.
- citations: every [arXiv:ID] is a real-looking id, and one of the papers this
  output may cite. In a review that holds by construction (code renders the
  ids), so a hit here means a bug, not a bad model: worth knowing either way.

What it doesn't catch: links written without http(s):// or www. ("evil.com"),
and paraphrased prompts. Detection is a bonus; least privilege is the net.
"""

import re
from collections.abc import Iterable
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from arxiv_agent.clients.arxiv import ARXIV_ID_PATTERN

ALLOWED_HOSTS = ("arxiv.org", "doi.org", "semanticscholar.org")
LINK = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.IGNORECASE)
# Citations are bracketed: [arXiv:2406.07791] or [arXiv:A, arXiv:B]. "Posted
# on arXiv: ..." in prose isn't one.
CITATION_GROUP = re.compile(r"\[(arXiv:[^\]]*)\]", re.IGNORECASE)
CITED_ID = re.compile(r"arXiv:\s*([^\s,;]+)", re.IGNORECASE)
VERSION = re.compile(r"v\d+$")  # 2406.07791v2 is the same paper as 2406.07791
WORD = re.compile(r"[a-z0-9]+")
LEAK_WORDS = 8


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: Literal["link", "prompt_leak", "citation"]
    detail: str


def _host_allowed(link: str) -> bool:
    host = urlparse(link if "://" in link else f"http://{link}").hostname or ""
    return any(
        host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_HOSTS
    )


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {
        tuple(words[i : i + LEAK_WORDS]) for i in range(len(words) - LEAK_WORDS + 1)
    }


class OutputGuard:
    def __init__(self, prompts: Iterable[str] = ()) -> None:
        # Every run of LEAK_WORDS words in our prompts, compared word by word
        # (case and punctuation ignored).
        self._prompt_shingles: set[tuple[str, ...]] = set()
        for prompt in prompts:
            self._prompt_shingles |= _shingles(prompt)

    def check(self, text: str, allowed_ids: set[str] | None = None) -> list[Violation]:
        # allowed_ids: the papers this text may cite; None skips the citation
        # check (an answer citing [S1]-style sources has none).
        violations = [
            Violation(
                check="link", detail=f"links to {link}, which isn't an allowed site"
            )
            for link in LINK.findall(text)
            if not _host_allowed(link)
        ]
        leaked = _shingles(text) & self._prompt_shingles
        if leaked:
            phrase = " ".join(min(leaked))
            violations.append(
                Violation(
                    check="prompt_leak",
                    detail=f"repeats its instructions: '{phrase} …'",
                )
            )
        if allowed_ids is not None:
            cited_ids = [
                cited
                for group in CITATION_GROUP.findall(text)
                for cited in CITED_ID.findall(group)
            ]
            for cited in cited_ids:
                if not ARXIV_ID_PATTERN.match(cited):
                    violations.append(
                        Violation(
                            check="citation", detail=f"cites '{cited}', not an arXiv id"
                        )
                    )
                elif VERSION.sub("", cited) not in allowed_ids:
                    violations.append(
                        Violation(
                            check="citation",
                            detail=f"cites arXiv:{cited}, which it wasn't allowed to cite",
                        )
                    )
        return violations


class Blocked(BaseModel):
    # A sentence the guard kept out of the output, and why.
    model_config = ConfigDict(extra="forbid")

    sentence: str
    violations: list[Violation]
