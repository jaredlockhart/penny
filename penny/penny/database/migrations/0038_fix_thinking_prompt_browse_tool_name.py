"""Fix unnotified-thoughts extraction_prompt: use explicit browse(queries=[...]) in step 3.

Type: data

The prompt seeded by migration 0033 for the ``unnotified-thoughts`` collection
instructs the Collector to browse the web in step 3, but uses a bare label
instead of explicit function-call syntax:

    3. browse — search the web and read one or two pages ...

All other steps in the same prompt use unambiguous function syntax:
    collection_read_random("likes", 1)
    read_latest("dislikes")
    collection_write("unnotified-thoughts", ...)
    done()

Without explicit call syntax, some models interpret "browse" as a description
of the action rather than the exact registered tool name.  Seeing "browser
extension" in the surrounding context and "search" in the step description,
they hallucinate ``browser.search`` — a non-existent tool — as the callable
name.  This produces ERROR-level "Tool not found: browser.search" log entries
and, when the model fixates on the wrong name, causes the entire thinking cycle
to abort without writing a thought.

This migration replaces the ambiguous step 3 label with explicit tool-call
syntax that matches the registered tool name and required parameter:
    3. browse(queries=["<seed topic>"]) — search the web and read one or two pages ...
"""

from __future__ import annotations

import sqlite3

_OLD = "3. browse — search the web"
_NEW = '3. browse(queries=["<seed topic>"]) — search the web'


def up(conn: sqlite3.Connection) -> None:
    """Fix step 3 of the unnotified-thoughts extraction_prompt."""
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, ?, ?) "
        "WHERE name = 'unnotified-thoughts' AND extraction_prompt LIKE ?",
        (_OLD, _NEW, f"%{_OLD}%"),
    )
    conn.commit()
