"""Backfill ``extraction_prompt`` for the system collections that have collectors.

Phase 4 of the per-collection collector refactor.  Three collections will be
populated by a unified ``CollectorAgent`` (one instance per collection,
scheduled in phase 5) instead of the bespoke preference / knowledge
extractors.  Their hand-tuned extraction prompts go on the memory row so
the scheduler can enumerate "collections that need a collector" by querying
``WHERE extraction_prompt IS NOT NULL``.

Three collections get prompts:

  - ``likes``     — positive preferences from user-messages
  - ``dislikes``  — negative preferences from user-messages
  - ``knowledge`` — page summaries from browse-results

System collections that aren't auto-curated leave ``extraction_prompt``
NULL: ``unnotified-thoughts`` and ``notified-thoughts`` (populated by
thinking + notify), ``user-profile`` (populated by chat / commands),
and the log memories (inputs, never outputs).

Idempotent — only writes when the row currently has NULL.

The legacy ``PREFERENCE_EXTRACTOR_SYSTEM_PROMPT`` covered both likes
and dislikes in a single agent run.  Splitting into per-valence
collectors gives each agent a narrower scope (the project's preferred
"focused narrow task" pattern) at the cost of one extra LLM call per
cycle reading the same input log — acceptable for the focus benefit.
"""

from __future__ import annotations

import sqlite3

_LIKES_PROMPT = (
    "You extract the user's positive preferences from their recent messages.\n\n"
    '1. Call log_read_next("user-messages") to fetch new messages you '
    "haven't seen yet.\n"
    "2. Identify every genuine LIKE — a thing the user wants, enjoys, "
    "seeks out, or expresses positive sentiment about.  Skip dislikes "
    "(a separate collector handles those), questions, factual statements, "
    "troubleshooting requests, and meta-instructions about Penny's "
    'behaviour ("remember this", "add to memory", "track this for me").\n'
    "3. For each like, the entry key is the topic itself, fully-qualified "
    "(3-10 words: 'Talk (album) by Yes', not 'the album'; 'single-origin "
    "coffee', not 'coffee stuff').  Reject vague or meta keys.  The "
    "content is the user's raw message that expressed the preference.\n"
    '4. Call collection_write("likes", entries=[...]) once with all '
    "extracted likes batched.\n"
    "5. If a recent message indicates an existing like is no longer "
    "accurate (e.g. 'I don't actually like X anymore'), call "
    'update_entry("likes", key=<topic>, content=<updated message>) '
    'or collection_delete_entry("likes", key=<topic>).\n'
    "6. Call done().\n\n"
    "Frustration about NOT FINDING something the user wants is a LIKE "
    "for that thing.  If no likes appear in the new messages, just call "
    "done() without writing anything."
)


_DISLIKES_PROMPT = (
    "You extract the user's negative preferences from their recent messages.\n\n"
    '1. Call log_read_next("user-messages") to fetch new messages you '
    "haven't seen yet.\n"
    "2. Identify every genuine DISLIKE — a thing the user avoids, "
    "complains about, or expresses negative sentiment toward.  Skip "
    "likes (a separate collector handles those), questions, factual "
    "statements, troubleshooting requests, and meta-instructions about "
    "Penny's behaviour.\n"
    "3. For each dislike, the entry key is the topic itself, "
    "fully-qualified (3-10 words: 'cilantro' not 'that herb'; "
    "'AI-generated music' not 'this stuff').  The content is the user's "
    "raw message that expressed the dislike.\n"
    '4. Call collection_write("dislikes", entries=[...]) once with all '
    "extracted dislikes batched.\n"
    "5. If a recent message indicates an existing dislike no longer "
    'applies, call update_entry("dislikes", key=<topic>, '
    'content=<updated message>) or collection_delete_entry("dislikes", key=<topic>).\n'
    "6. Call done().\n\n"
    "Frustration about NOT FINDING something the user wants is a LIKE "
    "for that thing, not a dislike — leave it for the likes collector. "
    "If no dislikes appear, just call done() without writing anything."
)


_KNOWLEDGE_PROMPT = (
    "You extract durable knowledge from web pages Penny has read.\n\n"
    '1. Call log_read_next("browse-results") to fetch new browse '
    "entries.  Each entry is one page (URL line, Title line, then "
    "page content).\n"
    "2. For each page entry, write a single dense paragraph of 8-12 "
    "sentences capturing the key factual content.  Focus on:\n"
    "   - What the thing IS (product, article, concept, etc.)\n"
    "   - Specific details that would be useful to recall later "
    "(specs, names, dates, claims, findings)\n"
    "   - What makes it notable or distinctive\n"
    "   Do NOT include navigation/ads/site chrome, "
    '"This page describes..." meta-framing, opinions about content '
    "quality, or anything not on the page.  Plain declarative "
    "prose; no bullets, no markdown, no headers.\n"
    '3. For each page, call collection_get("knowledge", key=<page '
    "title>) to see whether you already have a summary.  If one is "
    'returned, call update_entry("knowledge", key=<title>, '
    "content=<merged paragraph>) — integrate any new details from "
    "this fetch while preserving existing ones.  Otherwise, call "
    'collection_write("knowledge", entries=[{key: <title>, '
    "content: <new paragraph>}]).\n"
    "4. Call done().\n\n"
    "The entry's content should start with the page URL on its own "
    "line, then a blank line, then the summary paragraph — so "
    "retrieval can render the source link alongside the summary.\n\n"
    "If no new browse entries appear, call done() without writing "
    "anything."
)


def up(conn: sqlite3.Connection) -> None:
    rows = [
        ("likes", _LIKES_PROMPT),
        ("dislikes", _DISLIKES_PROMPT),
        ("knowledge", _KNOWLEDGE_PROMPT),
    ]
    for name, prompt in rows:
        conn.execute(
            "UPDATE memory SET extraction_prompt = ? WHERE name = ? AND extraction_prompt IS NULL",
            (prompt, name),
        )
    conn.commit()
