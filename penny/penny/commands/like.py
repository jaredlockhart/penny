"""/like command — list positive preferences or remove one by number."""

from __future__ import annotations

from penny.commands.preference_base import PreferenceListCommand
from penny.constants import PennyConstants


class LikeCommand(PreferenceListCommand):
    """List positive preferences or remove one by number."""

    name = "like"
    description = "Show your likes"
    help_text = (
        "Show or remove your liked preferences.\n\n"
        "**Usage**:\n"
        "• `/like` — Show numbered list of likes\n"
        "• `/like <number>` — Remove the like at that position\n\n"
        "**Examples**:\n"
        "• `/like`\n"
        "• `/like 2`"
    )
    valence = PennyConstants.PreferenceValence.POSITIVE
