"""/unlike command — list positive preferences or remove one by number."""

from __future__ import annotations

from penny.commands.preference_base import POSITIVE_CONFIG, PreferenceRemoveCommand


class UnlikeCommand(PreferenceRemoveCommand):
    """List positive preferences or remove one by number."""

    name = "unlike"
    description = "Remove a like"
    help_text = (
        "Show or remove your liked preferences.\n\n"
        "**Usage**:\n"
        "• `/unlike` — Show numbered list of likes\n"
        "• `/unlike <number>` — Remove the like at that position\n\n"
        "**Examples**:\n"
        "• `/unlike`\n"
        "• `/unlike 2`"
    )
    valence_config = POSITIVE_CONFIG
