"""Fix reactions stored without is_reaction flag.

_handle_reaction() was not passing is_reaction=True to log_message(), so all
reactions were stored as regular messages. This migration identifies them by
their emoji content + parent_id and sets the correct flag so the reaction
preference pipeline can pick them up.
"""

from __future__ import annotations

import sqlite3

# All recognized reaction emojis (positive + negative)
_REACTION_EMOJIS = (
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


def up(conn: sqlite3.Connection) -> None:
    """Fix reactions that were stored with is_reaction=0."""
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if "messagelog" not in tables:
        return

    placeholders = ",".join("?" for _ in _REACTION_EMOJIS)
    conn.execute(
        f"""
        UPDATE messagelog
        SET is_reaction = 1, processed = 0
        WHERE direction = 'incoming'
          AND parent_id IS NOT NULL
          AND is_reaction = 0
          AND content IN ({placeholders})
        """,
        _REACTION_EMOJIS,
    )
    conn.commit()
