"""Unify log reads onto a single ``log_read`` tool; fix notify + quality.

Type: data

The two log-read tools (``log_read_next`` cursor / ``log_read_recent`` window)
collapsed into one ``log_read`` whose behaviour is decided in Python from the
caller — collectors read cursor-batches (guaranteed coverage), chat/schedule
read a recent window.  The model no longer picks a read mode, so it can't choose
a window read for a review job and miss entries.

This migration brings the seeded extraction_prompts in line with the rename and
fixes two collectors:

1. Every extraction_prompt that named ``log_read_next(`` / ``log_read_recent(``
   now says ``log_read(`` (skills, likes, dislikes, knowledge, …). Pure rename —
   same behaviour, since a collector's ``log_read`` is cursor-mode.
2. The notify collector (``notified-thoughts``) drops its ``log_read_recent`` on
   ``penny-messages``.  That read was the *cause* of the duplicate-notification
   bug — the cycle re-sent what it read.  Dedup is structural: an unshared
   thought lives in ``unnotified-thoughts`` and ``collection_move`` removes it
   once sent, so it can never be re-picked.
3. The quality collector reads with ``log_read`` and now reviews EVERY entry in
   the batch (not just one) and fixes EACH drifted collection — the batch is
   bounded (``LOG_READ_LIMIT``) so this fits one cycle, and the cursor carries
   the rest to the next.
"""

from __future__ import annotations

import sqlite3

_NOTIFY_PROMPT = (
    "You are Penny's notify agent.  Once per cycle, share ONE fresh thought with "
    "your friend the user.\n\n"
    "Sequence:\n"
    '1. read_latest("unnotified-thoughts") — the thoughts you have NOT shared '
    "yet.  Sharing moves a thought out of this collection, so this list never "
    "contains anything you've already sent — you do not need to check past "
    "messages.\n"
    "2. Pick ONE that still seems interesting to the user.\n"
    "3. send_message(content=...) — deliver it conversationally: a greeting, all "
    "details from the thought (names, specs, dates), at least one source URL from "
    "the thought, and finish with an emoji.\n"
    '4. ONLY IF send_message returned "Message sent." then '
    'collection_move("unnotified-thoughts", "notified-thoughts", key=<chosen '
    "key>) — this marks it shared so it can never be picked again.  If the send "
    "failed, leave it in place.\n"
    "5. done().  If there's nothing fresh to share, just done()."
)

_QUALITY_PROMPT = (
    "You are Penny's quality agent.  Each cycle you review your own recent "
    "behaviour and fix EVERY collection that has drifted from what the user "
    "asked of it — then tell the user what you changed.\n\n"
    "A collection's `intent` is the user's own words for what it should do — the "
    "spec.  Its `extraction_prompt` is how it tries to do it.  When the prompt "
    "(or the behaviour it produces) no longer serves the intent, rewrite the "
    "prompt to match.  The intent is fixed — you can never change it; you change "
    "the prompt to honour it.\n\n"
    "Sequence:\n"
    '1. log_read("collector-runs") and log_read("penny-messages") — the next '
    "batch of what your collectors did and what you sent the user since you last "
    "reviewed.\n"
    "2. Review EVERY entry returned — do not stop at the first.  Flag each "
    "collection whose behaviour contradicts its stated intent: a message the "
    "user didn't ask for, the same thing sent twice, a silent collection that "
    "pinged.  If nothing in the batch looks wrong, call done() and change "
    "nothing — quiet batches are normal and expected.\n"
    "3. For EACH collection you flagged, fix it in turn:\n"
    "   a. collection_metadata(<collection>) — read its intent + current prompt.\n"
    "   b. Draft a corrected extraction_prompt: fix the offending step, keep "
    "every other step intact.  Unwanted pings (intent says stay silent): remove "
    "the send_message step.  Repeats: the offender is a step that reads your own "
    "past output and re-sends it — drop that read; not-repeating is handled by "
    "the collection's own move/write, never by re-sending.\n"
    "   c. prompt_test(collection=<collection>, extraction_prompt=<draft>) — "
    "dry-run it; if the cycle would still violate the intent, revise the draft "
    "and prompt_test again.  Only proceed once the dry run is clean.\n"
    "   d. collection_update(name=<collection>, extraction_prompt=<the "
    "dry-run-confirmed draft>).\n"
    "4. send_message the user one or two sentences naming each collection you "
    "fixed, what was going wrong, and what you changed.\n"
    "5. done().\n\n"
    "Only act on a clear, current contradiction between behaviour and intent.  "
    "Never weaken an intent to excuse a prompt."
)


def up(conn: sqlite3.Connection) -> None:
    # 1. Rename the read tool in every seeded/user extraction_prompt.
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, "
        "'log_read_next(', 'log_read(') WHERE extraction_prompt LIKE '%log_read_next(%'"
    )
    conn.execute(
        "UPDATE memory SET extraction_prompt = REPLACE(extraction_prompt, "
        "'log_read_recent(', 'log_read(') WHERE extraction_prompt LIKE '%log_read_recent(%'"
    )
    # 2. Notify → structural dedup (no penny-messages read).
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'notified-thoughts'",
        (_NOTIFY_PROMPT,),
    )
    # 3. Quality → review the whole batch, fix each drift.
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'quality'",
        (_QUALITY_PROMPT,),
    )
    conn.commit()
