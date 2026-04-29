"""Re-enable ambient recall for the ``penny-messages`` log.

PR #997 turned ``penny-messages`` recall off because every chat turn's
penny-side reply was already in the chat-turns ``history=`` array, so
the recall block was duplicating it.  But the chat-turns array is capped
at ``MESSAGE_CONTEXT_LIMIT`` recent messages — anything older was
invisible to ambient context, including conversations from previous
weeks the user later asks about.  Production confirmed this in a
"WeWorks you suggested" query that returned nothing despite seven
``penny-messages`` entries containing wework content.

Two later changes unblock flipping recall back on without re-introducing
the duplication problem:

  - PR #1006 added the self-match exclusion: the current turn's own
    text is filtered from the similarity corpus before scoring, so
    chat turns and recall context don't fight over the same anchor.
  - This PR adds the low-information filter to ``read_similar_hybrid``,
    which keeps stock greetings ("Hey!", "What's up?") from
    geometrically dominating the cosine ranking on short keyword
    anchors.

Together, ``recall=relevant`` on ``penny-messages`` now surfaces
historical Penny replies relevant to the current topic without
clobbering the recent conversation.

Idempotent — UPDATE is a no-op if the row is already ``relevant``.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET recall = 'relevant' WHERE name = 'penny-messages'",
    )
    conn.commit()
