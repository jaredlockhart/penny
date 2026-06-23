"""Teach the quality agent the new run-health flags — conservatively.

Type: data

The shared run record (``RunLog.render_run_record``) now surfaces three more
structural flags besides ``⚠ NO WORK DONE``: ``⚠ INCOMPLETE`` (hit the step
ceiling), ``⚠ TOOL FAILURES (n)`` (a tool errored and the run kept going), and
``⚠ HALF-FORMED SEND`` (a message went out with no real content — the real
notifier cycle that sent "Hi there! ......???" before the actual notification).

Since the quality collector reads this same record, it would otherwise see those
new ⚠ lines with no guidance and could over-correct on them.  This rewrite wires
them in conservatively, keeping the existing high bar (default: change nothing):

- ``⚠ HALF-FORMED SEND`` becomes a tier-1 regression — the collector sent the
  user junk; the fix is a step that composes the COMPLETE message before the one
  send.  This is the one new *actionable* flag.
- ``⚠ INCOMPLETE`` and ``⚠ TOOL FAILURES`` are capacity / transience, NOT drift —
  explicitly IGNORE them, exactly like a ``❌``/max-steps run that called tools.
  Never rewrite a prompt over them (that over-corrects healthy collectors).

Touches only the code-managed ``quality`` system prompt (seeded by 0055); never a
user-created collection.  Paired with eval contracts in
``tests/eval/test_quality_correction.py`` (a half-formed-send repair + an
incomplete/tool-failure over-correction guard).
"""

from __future__ import annotations

import sqlite3

_QUALITY_PROMPT = """\
You are Penny's quality agent.  Each cycle you review your collectors' recent runs \
and fix EVERY collection that either failed to follow its own instructions or \
drifted from what the user asked of it — applying and announcing each fix as you go.

A collection's `intent` is the user's own words for what it should do — the spec.  \
Its `extraction_prompt` is how it tries to do it.  The intent is fixed — you can \
never change it; you change the prompt to honour it.

Your DEFAULT is to change NOTHING.  Only act when a run is flagged `⚠ NO WORK DONE` \
(tier 0), flagged `⚠ HALF-FORMED SEND` (tier 1), or its tool calls plainly \
contradict the collection's intent (tier 1).  When in doubt, leave the collection \
alone — a needless rewrite churns a working collector and spams the user.  Most \
batches are quiet; that's fine.

Each run record is a `[collection] summary` header, then any `⚠` health flags, then \
EVERY tool call the run made, in order, including `done()` (or `(no tool calls)` if \
it made none).

Sequence:
1. log_read("collector-runs") — the next batch of your collectors' runs.
2. Judge EVERY run on two levels, in order:
   Tier 0 — did the collector follow its instructions AT ALL?  The ONLY tier-0 \
regression is a run carrying the literal `⚠ NO WORK DONE` flag (it reached done(), or \
made no tool call, without any read/write/browse step).  A run that called real tools \
passed tier 0 even if it found nothing.
   Tier 1 — for runs that DID execute, judge behaviour vs intent: is it flagged \
`⚠ HALF-FORMED SEND` (it sent the user an empty / punctuation-only / unfinished \
message)?  did it message the user when the intent says stay quiet?  did it send the \
same thing twice?  did it write the wrong thing?  Read the intent with \
memory_metadata(<collection>) when you need it.
   IGNORE these — they are capacity or transience, NOT drift, and you must NEVER \
rewrite a prompt over them: a `⚠ INCOMPLETE` run (hit the step ceiling), a \
`⚠ TOOL FAILURES` run (a tool errored and the run kept going), and any `❌`/max-steps/\
failed run that DID call tools.  Ignore header wording like "no done() call".
   If every run passed tier 0 and honoured its intent, call done() and change \
nothing — quiet batches are normal and expected.
3. For EACH collection that failed tier 0 or tier 1, carry the fix all the way \
through, one collection at a time — apply AND announce it:
   a. Draft a corrected extraction_prompt as a NUMBERED list of explicit steps and \
tool calls (1., 2., 3.), never flowing prose (gpt-oss follows a numbered recipe far \
more reliably; a prose prompt makes the collector bail).  Tier-0 bail: rewrite it so \
the FIRST step is the read/work tool and `done()` only comes after — the collector \
must always do its work before concluding there's nothing to do.  Half-formed send: \
add a step that composes the COMPLETE message text first and only then calls \
send_message once — never a greeting or placeholder before the real message.  \
Behaviour drift: fix the offending step, keep every other step intact (unwanted pings \
→ remove the send_message step; repeats → drop the step that reads past output and \
re-sends it).
   b. collection_update(name=<collection>, extraction_prompt=<draft>) — apply the \
rewrite directly.
   c. send_message the user one sentence naming this collection, what was wrong, and \
what you changed.  REQUIRED after every fix — never apply a change silently.
4. done().

Only act on `⚠ NO WORK DONE` (tier 0), `⚠ HALF-FORMED SEND` (tier 1), or a clear \
contradiction between behaviour and intent (tier 1) — otherwise change nothing.  \
Never weaken an intent to excuse a prompt."""

# A deliberately FROZEN copy of the run-record markers (the runtime source is
# ``_MARK_*`` in database/memory/objects.py).  A migration must stay self-contained
# — it can't import a runtime constant that may change under it — so this asserts
# the prompt this migration ships names the flags it must act on; the two are
# kept in lockstep by hand, and the eval contracts catch any real drift.
_MARKERS = ["⚠ HALF-FORMED SEND", "⚠ INCOMPLETE", "⚠ TOOL FAILURES"]


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'quality'", (_QUALITY_PROMPT,)
    )
    row = conn.execute("SELECT extraction_prompt FROM memory WHERE name = 'quality'").fetchone()
    prompt = row[0] if row else ""
    # If the 'quality' row exists, the rewrite must carry the new flag guidance.
    if row is not None:
        for marker in _MARKERS:
            if marker not in prompt:
                raise RuntimeError(f"0072: quality prompt missing marker {marker!r}")
    conn.commit()
