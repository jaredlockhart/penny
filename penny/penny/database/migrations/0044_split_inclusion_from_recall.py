"""Split memory recall into two orthogonal stages: inclusion + recall.

The single ``recall`` flag conflated two decisions the new two-stage recall
pipeline separates:

  * Stage 1 — ``inclusion`` (``always`` / ``relevant`` / ``never``): does this
    memory feed ambient recall at all (collection routing).
  * Stage 2 — ``recall`` (``all`` / ``relevant`` / ``recent``): which of its
    entries surface once included (entry rendering).

This migration adds the ``inclusion`` and ``description_embedding`` columns,
derives ``inclusion`` from the old ``recall`` value, then collapses the old
``recall=off`` rows onto a valid entry mode:

  old recall   ->  inclusion   recall
  ----------       ---------   ------
  off              never       recent   (never included; mode is moot)
  recent           always      recent
  all              always      all
  relevant         relevant    relevant

Finally, memories that should always be available regardless of topic are
forced to ``inclusion=always``:

  * ``skills`` — workflow recipes score low on absolute cosine and must never
    be gated out.
  * ``user-messages`` / ``penny-messages`` — a topical description anchor
    would wrongly drop conversation context on topic-less turns.
  * ``user-profile`` — the user's own identity is relevant to every turn.
  * ``likes`` / ``dislikes`` / ``knowledge`` — these are structural containers
    whose description is *meta* ("topics the user likes", "facts from web
    pages"), not topical, so the stage-1 description anchor scores near-zero
    against any specific conversation and would gate them out exactly when a
    relevant entry exists.  They stay always-in; stage-2 entry ranking does
    the topical filtering instead.

Genuinely topical collections (a user's research collections — bars in a
city, credit-card options, etc.) keep ``inclusion=relevant``: their
description *is* the topic, so the anchor gate works as intended.

``description_embedding`` is left NULL; the startup backfill vectorizes each
active description (migrations can't call the embedding model).
"""

from __future__ import annotations

import sqlite3

# Memories whose relevance is unconditional — never gate them behind the
# stage-1 description anchor.  Includes the structural containers
# (likes / dislikes / knowledge) whose meta description can't anchor a
# topical match; stage-2 entry ranking filters those instead.
_ALWAYS = (
    "skills",
    "user-messages",
    "penny-messages",
    "user-profile",
    "likes",
    "dislikes",
    "knowledge",
)


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """True if ``table`` already has ``column`` — keeps ADD COLUMN idempotent.

    Test DBs build the schema from the current SQLModel models (which already
    declare ``inclusion`` / ``description_embedding``); production upgrades a
    pre-existing table that has neither.  Guarding both covers both paths.
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def up(conn: sqlite3.Connection) -> None:
    if not _has_column(conn, "memory", "inclusion"):
        conn.execute("ALTER TABLE memory ADD COLUMN inclusion TEXT NOT NULL DEFAULT 'relevant'")
    if not _has_column(conn, "memory", "description_embedding"):
        conn.execute("ALTER TABLE memory ADD COLUMN description_embedding BLOB")

    # Derive inclusion from the old recall value (before rewriting recall).
    conn.execute("UPDATE memory SET inclusion = 'never'  WHERE recall = 'off'")
    conn.execute("UPDATE memory SET inclusion = 'always' WHERE recall IN ('recent', 'all')")
    conn.execute("UPDATE memory SET inclusion = 'relevant' WHERE recall = 'relevant'")

    # Collapse the retired 'off' entry mode onto a valid one.
    conn.execute("UPDATE memory SET recall = 'recent' WHERE recall = 'off'")

    # Force unconditional inclusion for identity / conversation / skills.
    placeholders = ", ".join("?" for _ in _ALWAYS)
    conn.execute(
        f"UPDATE memory SET inclusion = 'always' WHERE name IN ({placeholders})",
        _ALWAYS,
    )
    conn.commit()
