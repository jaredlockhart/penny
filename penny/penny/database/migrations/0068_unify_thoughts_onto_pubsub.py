"""Unify the thoughts pipeline onto pub/sub.

Type: data

Replaces the bespoke ``unnotified-thoughts`` (producer) → ``notified-thoughts``
(move-drain consumer) pair with a single ``thoughts`` producer that the generic
``notifier`` (migration 0067) drains by cursor — the same model every other
research collection uses.  Notification state is now the notifier's forward-only
cursor, not which of two collections an entry sits in, so there's no move.

Steps:
1. Create ``thoughts`` — the thinking producer, ``published=true`` so the notifier
   delivers each new thought once; ``inclusion=relevant`` so past thoughts still
   surface in chat (as ``notified-thoughts`` did).  The prompt is the old
   ``unnotified-thoughts`` thinking prompt with its dedup + write retargeted from
   the two old collections to ``thoughts`` itself.
2. Move every existing thought (both old collections) into ``thoughts``.
3. Seed the notifier's cursor for ``thoughts`` to the newest moved entry, so the
   cutover doesn't re-deliver the ~hundreds of thoughts already shared.
4. Archive the two old collections (data preserved, no longer dispatched).

After this, nothing uses ``collection_move`` (the only caller was
``notified-thoughts``); the tool is removed in the same PR.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

THOUGHTS_DESCRIPTION = "Penny's inner-monologue thoughts about the user's interests."
THOUGHTS_INTENT = (
    "Your goal is to pick a random topic from the things you've collected in the likes "
    "collection, learn more about it, and store what you find."
)
THOUGHTS_PROMPT = (
    "You are Penny's thinking agent. Once per run, you find ONE specific, concrete thing "
    "worth knowing about — something the user would enjoy hearing — and store it as a "
    "thought.\n"
    "\n"
    "Sequence:\n"
    '1. collection_read_random("likes", 1) — pick one seed topic from the user\'s likes.\n'
    '2. collection_read_latest("dislikes") — see what the user doesn\'t like.\n'
    '3. browse(queries=["<seed topic>"]) — search the web and read one or two pages to find '
    "something timely and interesting grounded in the seed topic.\n"
    "4. Draft ONE thought connecting what you found to the seed.  Write it conversationally, "
    "like you're texting a friend; include specific details (names, specs, dates), at least "
    "one source URL, and finish with an emoji.  Keep it under 300 words.\n"
    "5. Check the draft against the dislikes list.  If it conflicts with anything the user "
    "dislikes, call done() without writing.\n"
    '6. exists(["thoughts"], key, content) — if a similar thought already exists, call done() '
    "without writing.\n"
    '7. collection_write("thoughts", entries=[{key: short topic name (3-10 words), content: '
    "the thought you drafted}]).\n"
    "8. done().\n"
    "\n"
    "The interesting stuff is ON the pages, not in search snippets — browse the URLs you find "
    "rather than searching forever.  If nothing noteworthy comes up, call done() without "
    "writing; quiet cycles are normal.  Troubleshooting guides, bug workarounds, and support "
    "articles are NOT interesting discoveries."
)

_INTERVAL_SECONDS = 5400  # the thinking cadence, from unnotified-thoughts


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    now = datetime.now(UTC).isoformat()

    # 1. The unified producer.
    conn.execute(
        "INSERT OR IGNORE INTO memory "
        "(name, type, description, inclusion, recall, archived, created_at, "
        "extraction_prompt, collector_interval_seconds, base_interval_seconds, "
        "intent, published) "
        "VALUES ('thoughts', 'collection', ?, 'relevant', 'relevant', 0, ?, ?, ?, ?, ?, 1)",
        (
            THOUGHTS_DESCRIPTION,
            now,
            THOUGHTS_PROMPT,
            _INTERVAL_SECONDS,
            _INTERVAL_SECONDS,
            THOUGHTS_INTENT,
        ),
    )

    # 2. Move every existing thought into the unified collection (keys may repeat
    #    across the two sources — collections dedup on write, not via a DB
    #    constraint, so the move is safe).
    conn.execute(
        "UPDATE memory_entry SET memory_name = 'thoughts' "
        "WHERE memory_name IN ('unnotified-thoughts', 'notified-thoughts')"
    )

    # 3. Seed the notifier's cursor to the newest moved thought so the backlog
    #    (already delivered the old way) isn't re-sent — only thoughts written
    #    after the cutover get delivered.  Skip on a fresh instance with none.
    conn.execute(
        "INSERT OR IGNORE INTO agent_cursor (agent_name, memory_name, last_read_at, updated_at) "
        "SELECT 'notifier', 'thoughts', MAX(created_at), ? "
        "FROM memory_entry WHERE memory_name = 'thoughts' "
        "HAVING MAX(created_at) IS NOT NULL",
        (now,),
    )

    # 4. Retire the old collections (data already moved out; keep as archived shells).
    conn.execute(
        "UPDATE memory SET archived = 1 WHERE name IN ('unnotified-thoughts', 'notified-thoughts')"
    )
    conn.commit()
