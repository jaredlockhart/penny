"""PreferenceExtractorAgent — extracts user likes and dislikes.

Reads new entries from ``user-messages`` via ``log_read_next``,
identifies preferences across the returned messages, and writes
them to the ``likes`` / ``dislikes`` collections.  Cursor is
committed only on a clean ``done()`` exit so a failed run replays
on the next schedule.
"""

from __future__ import annotations

from penny.agents.base import Agent
from penny.constants import HistoryPromptType


class PreferenceExtractorAgent(Agent):
    """Background worker that extracts preferences from user messages."""

    name = "preference-extractor"
    prompt_type = HistoryPromptType.PREFERENCE_EXTRACTION

    # Cap on agentic loop iterations.  The expected flow is
    # read_next → write(likes) → write(dislikes) → done, so 8 leaves
    # headroom for re-reads or batched writes without letting a
    # runaway loop tail through forever.
    MAX_STEPS = 8

    def get_max_steps(self) -> int:
        return self.MAX_STEPS
