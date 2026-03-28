"""Add valence column to thought table for user reaction tracking."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Add valence column to thought and backfill from existing reaction messages.

    Backfill logic:
      - Find messagelog rows where is_reaction=1 and parent_id points to a
        message whose thought_id is not NULL.
      - Set thought.valence = 1 for positive emojis, -1 for negative emojis.
      - Only sets valence when the thought currently has NULL valence (does not
        overwrite a later reaction).

    Emoji sets are duplicated from PennyConstants to keep migrations self-contained.
    """
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='thought'"
        ).fetchall()
    ]
    if not tables:
        conn.commit()
        return

    columns = [row[1] for row in conn.execute("PRAGMA table_info(thought)").fetchall()]
    if "valence" not in columns:
        conn.execute("ALTER TABLE thought ADD COLUMN valence INTEGER")

    positive_emojis = (
        "\U0001f44d",  # 👍
        "\u2764\ufe0f",  # ❤️
        "\U0001f525",  # 🔥
        "\U0001f44f",  # 👏
        "\U0001f60d",  # 😍
        "\U0001f64c",  # 🙌
        "\U0001f4af",  # 💯
        "\u2b50",  # ⭐
        "\U0001f60a",  # 😊
        "\U0001f389",  # 🎉
        "\U0001f4aa",  # 💪
        "\u2705",  # ✅
        "\U0001f929",  # 🤩
    )
    negative_emojis = (
        "\U0001f44e",  # 👎
        "\U0001f621",  # 😡
        "\U0001f92e",  # 🤮
        "\U0001f4a9",  # 💩
        "\U0001f624",  # 😤
        "\u274c",  # ❌
        "\U0001f61e",  # 😞
        "\U0001f612",  # 😒
        "\U0001f644",  # 🙄
    )

    pos_placeholders = ",".join("?" * len(positive_emojis))
    neg_placeholders = ",".join("?" * len(negative_emojis))

    conn.execute(
        f"""
        UPDATE thought SET valence = 1
        WHERE id IN (
            SELECT parent.thought_id
            FROM messagelog AS reaction
            JOIN messagelog AS parent ON reaction.parent_id = parent.id
            WHERE reaction.is_reaction = 1
              AND reaction.content IN ({pos_placeholders})
              AND parent.thought_id IS NOT NULL
        ) AND valence IS NULL
        """,
        list(positive_emojis),
    )

    conn.execute(
        f"""
        UPDATE thought SET valence = -1
        WHERE id IN (
            SELECT parent.thought_id
            FROM messagelog AS reaction
            JOIN messagelog AS parent ON reaction.parent_id = parent.id
            WHERE reaction.is_reaction = 1
              AND reaction.content IN ({neg_placeholders})
              AND parent.thought_id IS NOT NULL
        ) AND valence IS NULL
        """,
        list(negative_emojis),
    )

    conn.commit()
