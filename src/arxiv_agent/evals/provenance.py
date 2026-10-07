"""What an eval run stamps on its results, so a score can be traced to the code
and data that produced it."""

import hashlib
import subprocess
from pathlib import Path


def git_version() -> str:
    # A score is only reproducible from committed code, so say when it wasn't.
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return f"{sha or 'unknown'}{'-dirty' if dirty else ''}"


def content_hash(path: Path) -> str:
    # Hash the text with normalized line endings, so the same eval set gets the
    # same hash on Windows (CRLF) and Linux (LF).
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
