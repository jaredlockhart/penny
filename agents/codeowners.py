"""CODEOWNERS parser for trusted user identification.

Parses GitHub CODEOWNERS files to extract trusted maintainer usernames.
Used by the agent orchestrator to filter issue content before passing
it to Claude CLI agents.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CODEOWNERS_PATHS = [
    ".github/CODEOWNERS",
    "CODEOWNERS",
    "docs/CODEOWNERS",
]


def parse_codeowners(project_root: Path) -> set[str]:
    """Parse CODEOWNERS file and return set of trusted GitHub usernames.

    Searches standard CODEOWNERS locations. Extracts @username tokens,
    ignoring @org/team references (which contain a slash).

    Returns empty set if no CODEOWNERS file found.
    """
    for relative_path in CODEOWNERS_PATHS:
        codeowners_path = project_root / relative_path
        if codeowners_path.is_file():
            return _parse_file(codeowners_path)

    logger.warning("No CODEOWNERS file found in standard locations")
    return set()


def _parse_file(path: Path) -> set[str]:
    """Extract unique GitHub usernames from a CODEOWNERS file."""
    usernames: set[str] = set()

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # CODEOWNERS format: <pattern> @user1 @user2 ...
        # Tokens after the file pattern are owners
        tokens = line.split()
        for token in tokens:
            if token.startswith("@") and "/" not in token:
                usernames.add(token.lstrip("@"))

    logger.info(f"Loaded {len(usernames)} trusted user(s) from CODEOWNERS: {usernames}")
    return usernames
