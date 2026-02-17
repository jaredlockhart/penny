"""Preference commands — /like, /dislike, /unlike, /undislike."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class LikeCommand(Command):
    """View or add likes."""

    name = "like"
    description = "View or add things you like"
    help_text = (
        "View your stored likes or add a new like.\n\n"
        "**Usage**:\n"
        "- `/like` — View all your likes\n"
        "- `/like <topic>` — Add a topic to your likes\n\n"
        "**Examples**:\n"
        "- `/like`\n"
        "- `/like video games`\n"
        "- `/like sci-fi movies`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute like command."""
        args = args.strip()

        # No args - list all likes
        if not args:
            likes = context.db.get_preferences(context.user, PennyConstants.PreferenceType.LIKE)
            if not likes:
                return CommandResult(text=PennyResponse.LIKES_EMPTY)

            lines = [PennyResponse.LIKES_HEADER, ""]
            for i, pref in enumerate(likes, 1):
                lines.append(f"{i}. {pref.topic}")
            return CommandResult(text="\n".join(lines))

        # Add new like
        topic = args

        # Check for conflicts in dislikes
        conflict = context.db.find_conflicting_preference(
            context.user, topic, PennyConstants.PreferenceType.LIKE
        )

        if conflict:
            # Remove from dislikes
            context.db.remove_preference(context.user, topic, PennyConstants.PreferenceType.DISLIKE)
            # Add to likes
            context.db.add_preference(context.user, topic, PennyConstants.PreferenceType.LIKE)
            return CommandResult(text=PennyResponse.LIKE_ADDED_CONFLICT.format(topic=topic))
        else:
            # Just add to likes
            added = context.db.add_preference(
                context.user, topic, PennyConstants.PreferenceType.LIKE
            )
            if added:
                return CommandResult(text=PennyResponse.LIKE_ADDED.format(topic=topic))
            else:
                return CommandResult(text=PennyResponse.LIKE_DUPLICATE.format(topic=topic))


class DislikeCommand(Command):
    """View or add dislikes."""

    name = "dislike"
    description = "View or add things you dislike"
    help_text = (
        "View your stored dislikes or add a new dislike.\n\n"
        "**Usage**:\n"
        "- `/dislike` — View all your dislikes\n"
        "- `/dislike <topic>` — Add a topic to your dislikes\n\n"
        "**Examples**:\n"
        "- `/dislike`\n"
        "- `/dislike ai music`\n"
        "- `/dislike bananas 🍌`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute dislike command."""
        args = args.strip()

        # No args - list all dislikes
        if not args:
            dislikes = context.db.get_preferences(
                context.user, PennyConstants.PreferenceType.DISLIKE
            )
            if not dislikes:
                return CommandResult(text=PennyResponse.DISLIKES_EMPTY)

            lines = [PennyResponse.DISLIKES_HEADER, ""]
            for i, pref in enumerate(dislikes, 1):
                lines.append(f"{i}. {pref.topic}")
            return CommandResult(text="\n".join(lines))

        # Add new dislike
        topic = args

        # Check for conflicts in likes
        conflict = context.db.find_conflicting_preference(
            context.user, topic, PennyConstants.PreferenceType.DISLIKE
        )

        if conflict:
            # Remove from likes
            context.db.remove_preference(context.user, topic, PennyConstants.PreferenceType.LIKE)
            # Add to dislikes
            context.db.add_preference(context.user, topic, PennyConstants.PreferenceType.DISLIKE)
            return CommandResult(text=PennyResponse.DISLIKE_ADDED_CONFLICT.format(topic=topic))
        else:
            # Just add to dislikes
            added = context.db.add_preference(
                context.user, topic, PennyConstants.PreferenceType.DISLIKE
            )
            if added:
                return CommandResult(text=PennyResponse.DISLIKE_ADDED.format(topic=topic))
            else:
                return CommandResult(text=PennyResponse.DISLIKE_DUPLICATE.format(topic=topic))


class UnlikeCommand(Command):
    """Remove a like."""

    name = "unlike"
    description = "Remove something from your likes"
    help_text = (
        "Remove a topic from your likes.\n\n"
        "**Usage**:\n"
        "- `/unlike <topic>` — Remove a topic from your likes\n"
        "- `/unlike <number>` — Remove a topic by its list position\n\n"
        "**Examples**:\n"
        "- `/unlike video games`\n"
        "- `/unlike 2`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute unlike command."""
        args = args.strip()

        if not args:
            return CommandResult(text=PennyResponse.UNLIKE_USAGE)

        # Check if args is a number (list position)
        if args.isdigit():
            position = int(args)
            likes = context.db.get_preferences(context.user, PennyConstants.PreferenceType.LIKE)

            if position < 1 or position > len(likes):
                return CommandResult(
                    text=PennyResponse.UNLIKE_OUT_OF_RANGE.format(position=position)
                )

            # Get the topic at this position (1-indexed)
            topic = likes[position - 1].topic
        else:
            # Use the full topic string
            topic = args

        removed = context.db.remove_preference(
            context.user, topic, PennyConstants.PreferenceType.LIKE
        )

        if removed:
            return CommandResult(text=PennyResponse.UNLIKE_REMOVED.format(topic=topic))
        else:
            return CommandResult(text=PennyResponse.UNLIKE_NOT_FOUND.format(topic=topic))


class UndislikeCommand(Command):
    """Remove a dislike."""

    name = "undislike"
    description = "Remove something from your dislikes"
    help_text = (
        "Remove a topic from your dislikes.\n\n"
        "**Usage**:\n"
        "- `/undislike <topic>` — Remove a topic from your dislikes\n"
        "- `/undislike <number>` — Remove a topic by its list position\n\n"
        "**Examples**:\n"
        "- `/undislike bananas`\n"
        "- `/undislike 1`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute undislike command."""
        args = args.strip()

        if not args:
            return CommandResult(text=PennyResponse.UNDISLIKE_USAGE)

        # Check if args is a number (list position)
        if args.isdigit():
            position = int(args)
            dislikes = context.db.get_preferences(
                context.user, PennyConstants.PreferenceType.DISLIKE
            )

            if position < 1 or position > len(dislikes):
                return CommandResult(
                    text=PennyResponse.UNDISLIKE_OUT_OF_RANGE.format(position=position)
                )

            # Get the topic at this position (1-indexed)
            topic = dislikes[position - 1].topic
        else:
            # Use the full topic string
            topic = args

        removed = context.db.remove_preference(
            context.user, topic, PennyConstants.PreferenceType.DISLIKE
        )

        if removed:
            return CommandResult(text=PennyResponse.UNDISLIKE_REMOVED.format(topic=topic))
        else:
            return CommandResult(text=PennyResponse.UNDISLIKE_NOT_FOUND.format(topic=topic))
