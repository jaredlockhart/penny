"""Preference commands — /like, /dislike, /unlike, /undislike."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.database.models import Preference
from penny.ollama.embeddings import deserialize_embedding, find_similar
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)

# Similarity threshold for matching entities to preference topics
_ENTITY_MATCH_THRESHOLD = 0.4
_ENTITY_MATCH_TOP_K = 10


async def _record_preference_engagements(
    context: CommandContext,
    preference: Preference,
    engagement_type: str,
    valence: str,
    strength: float,
) -> None:
    """Record engagements for a preference command (like or dislike).

    Creates a preference-level engagement, then tries to find matching
    entities via embedding similarity and records entity-level engagements
    for each match. Entity matching is best-effort — fails silently if
    no embedding model is configured or Ollama is unavailable.
    """
    assert preference.id is not None

    # Record preference-level engagement
    context.db.add_engagement(
        user=context.user,
        engagement_type=engagement_type,
        valence=valence,
        strength=strength,
        preference_id=preference.id,
    )

    # Try to find matching entities via embedding similarity
    embedding_model = context.config.ollama_embedding_model
    if not embedding_model:
        return

    try:
        entities = context.db.get_user_entities_with_embeddings(context.user)
        if not entities:
            return

        # Embed the preference topic
        vecs = await context.ollama_client.embed(preference.topic, model=embedding_model)
        query_embedding = vecs[0]

        # Build candidates from entities with embeddings
        candidates = []
        for entity in entities:
            assert entity.id is not None
            assert entity.embedding is not None
            candidates.append((entity.id, deserialize_embedding(entity.embedding)))

        matches = find_similar(
            query_embedding,
            candidates,
            top_k=_ENTITY_MATCH_TOP_K,
            threshold=_ENTITY_MATCH_THRESHOLD,
        )

        for entity_id, _score in matches:
            context.db.add_engagement(
                user=context.user,
                engagement_type=engagement_type,
                valence=valence,
                strength=strength,
                entity_id=entity_id,
                preference_id=preference.id,
            )
    except Exception:
        logger.debug("Entity matching skipped for preference '%s'", preference.topic, exc_info=True)


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
            pref = context.db.add_preference(
                context.user, topic, PennyConstants.PreferenceType.LIKE
            )
            if pref:
                await _record_preference_engagements(
                    context,
                    pref,
                    PennyConstants.EngagementType.LIKE_COMMAND,
                    PennyConstants.EngagementValence.POSITIVE,
                    PennyConstants.ENGAGEMENT_STRENGTH_LIKE_COMMAND,
                )
            return CommandResult(text=PennyResponse.LIKE_ADDED_CONFLICT.format(topic=topic))
        else:
            # Just add to likes
            pref = context.db.add_preference(
                context.user, topic, PennyConstants.PreferenceType.LIKE
            )
            if pref:
                await _record_preference_engagements(
                    context,
                    pref,
                    PennyConstants.EngagementType.LIKE_COMMAND,
                    PennyConstants.EngagementValence.POSITIVE,
                    PennyConstants.ENGAGEMENT_STRENGTH_LIKE_COMMAND,
                )
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
            pref = context.db.add_preference(
                context.user, topic, PennyConstants.PreferenceType.DISLIKE
            )
            if pref:
                await _record_preference_engagements(
                    context,
                    pref,
                    PennyConstants.EngagementType.DISLIKE_COMMAND,
                    PennyConstants.EngagementValence.NEGATIVE,
                    PennyConstants.ENGAGEMENT_STRENGTH_DISLIKE_COMMAND,
                )
            return CommandResult(text=PennyResponse.DISLIKE_ADDED_CONFLICT.format(topic=topic))
        else:
            # Just add to dislikes
            pref = context.db.add_preference(
                context.user, topic, PennyConstants.PreferenceType.DISLIKE
            )
            if pref:
                await _record_preference_engagements(
                    context,
                    pref,
                    PennyConstants.EngagementType.DISLIKE_COMMAND,
                    PennyConstants.EngagementValence.NEGATIVE,
                    PennyConstants.ENGAGEMENT_STRENGTH_DISLIKE_COMMAND,
                )
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
