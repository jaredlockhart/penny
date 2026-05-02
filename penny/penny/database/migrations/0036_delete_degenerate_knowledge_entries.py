"""Delete degenerate entries from the ``knowledge`` memory collection.

Removes entries whose ``content`` would now be rejected by the write-time
guard added in PR #1020:

  - No word tokens (empty, pure punctuation, ellipsis)
  - Bare URL with no descriptive text
  - Known LLM bail-out phrases ('Not sure', 'I cannot help with that', etc.)

These entries pollute relevant-mode recall, appearing at the top of
similarity rankings for almost every query because their embeddings sit
near the corpus centroid.  Write-time guards prevent new ones; this
migration removes the historical pollution from the prod corpus.

Type: data
"""

from __future__ import annotations

import re
import sqlite3

_WORD_TOKEN_RE = re.compile(r"\w+")
_BARE_URL_RE = re.compile(r"^https?://\S+$")

_BAILOUT_PHRASES: frozenset[str] = frozenset(
    {
        "not sure",
        "i'm not sure",
        "i am not sure",
        "i cannot help with that",
        "i can't help with that",
        "i don't know",
        "i do not know",
        "n/a",
        "no information",
        "no information available",
        "unable to summarize",
        "unable to provide a summary",
        "no content available",
        "content not available",
        "page not available",
        "content unavailable",
        "access denied",
        "error",
    }
)


def _is_degenerate(content: str) -> bool:
    stripped = content.strip()
    if not _WORD_TOKEN_RE.findall(stripped):
        return True
    if _BARE_URL_RE.match(stripped):
        return True
    return stripped.lower() in _BAILOUT_PHRASES


def up(conn: sqlite3.Connection) -> None:
    """Delete degenerate entries from the knowledge collection."""
    rows = conn.execute(
        "SELECT id, content FROM memory_entry WHERE memory_name = 'knowledge'"
    ).fetchall()

    ids_to_delete = [row[0] for row in rows if _is_degenerate(row[1])]

    if ids_to_delete:
        placeholders = ",".join("?" * len(ids_to_delete))
        conn.execute(
            f"DELETE FROM memory_entry WHERE id IN ({placeholders})",
            ids_to_delete,
        )

    conn.commit()
