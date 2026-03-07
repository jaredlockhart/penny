"""/dislike command — list negative preferences or remove one by number."""

from __future__ import annotations

from penny.commands.preference_base import PreferenceListCommand
from penny.constants import PennyConstants


class DislikeCommand(PreferenceListCommand):
    """List negative preferences or remove one by number."""

    name = "dislike"
    description = "Show your dislikes"
    help_text = (
        "Show or remove your disliked preferences.\n\n"
        "**Usage**:\n"
        "• `/dislike` — Show numbered list of dislikes\n"
        "• `/dislike <number>` — Remove the dislike at that position\n\n"
        "**Examples**:\n"
        "• `/dislike`\n"
        "• `/dislike 2`"
    )
    valence = PennyConstants.PreferenceValence.NEGATIVE
