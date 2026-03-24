"""/undislike command — list negative preferences or remove one by number."""

from __future__ import annotations

from penny.commands.preference_base import NEGATIVE_CONFIG, PreferenceRemoveCommand


class UndislikeCommand(PreferenceRemoveCommand):
    """List negative preferences or remove one by number."""

    name = "undislike"
    description = "Remove a dislike"
    help_text = (
        "Show or remove your disliked preferences.\n\n"
        "**Usage**:\n"
        "• `/undislike` — Show numbered list of dislikes\n"
        "• `/undislike <number>` — Remove the dislike at that position\n\n"
        "**Examples**:\n"
        "• `/undislike`\n"
        "• `/undislike 2`"
    )
    valence_config = NEGATIVE_CONFIG
