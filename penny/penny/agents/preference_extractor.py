"""PreferenceExtractorAgent — extracts user likes and dislikes.

Reads new entries from ``user-messages`` via ``log_read_next``,
identifies preferences across the returned messages, and writes
them to the ``likes`` / ``dislikes`` collections.  Cursor is
committed only on a clean ``done()`` exit so a failed run replays
on the next schedule.
"""

from __future__ import annotations

from penny.agents.base import Agent
from penny.prompts import Prompt


class PreferenceExtractorAgent(Agent):
    """Background worker that extracts preferences from user messages."""

    name = "preference-extractor"
    system_prompt = Prompt.PREFERENCE_EXTRACTOR_SYSTEM_PROMPT
