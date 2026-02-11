"""PreferenceAgent for extracting user preferences from messages and reactions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import DISLIKE_REACTIONS, LIKE_REACTIONS, PreferenceType

if TYPE_CHECKING:
    from penny.channels import MessageChannel
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


class ReactionAnalysis(BaseModel):
    """Schema for parsing reaction analysis from LLM."""

    topic: str = Field(
        description="The main topic or subject of the message (a short phrase or keyword)"
    )


class ExtractedPreference(BaseModel):
    """A single preference extracted from messages."""

    topic: str = Field(description="Short phrase or keyword (1-4 words)")
    type: str = Field(description="'like' or 'dislike'")


class MessagePreferences(BaseModel):
    """Schema for batch preference extraction from messages."""

    preferences: list[ExtractedPreference] = Field(
        default_factory=list,
        description="List of new preferences found in the messages",
    )


class PreferenceAgent(Agent):
    """Agent for extracting user preferences from messages and reactions."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "preference"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending notifications."""
        self._channel = channel

    async def execute(self) -> bool:
        """
        Process preference updates from reactions and messages.

        Returns:
            True if work was done, False if nothing to do
        """
        if not self._channel:
            logger.warning("PreferenceAgent: no channel set, skipping")
            return False

        reaction_work = await self._analyze_reactions()
        message_work = await self._analyze_messages()
        return reaction_work or message_work

    async def _analyze_reactions(self) -> bool:
        """
        Analyze recent reactions and update preferences.

        Returns:
            True if reactions were processed, False if nothing to do
        """
        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False
        for sender in senders:
            reactions = self.db.get_user_reactions(sender, limit=10)
            if not reactions:
                continue

            for reaction_msg in reactions:
                processed = await self._process_reaction(sender, reaction_msg)
                # Mark as processed regardless of whether we updated preferences
                if reaction_msg.id is not None:
                    self.db.mark_reaction_processed(reaction_msg.id)
                if processed:
                    work_done = True

        return work_done

    async def _analyze_messages(self) -> bool:
        """
        Analyze unprocessed user messages and extract preferences.

        Returns:
            True if preferences were updated, False if nothing to do
        """
        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False
        for sender in senders:
            messages = self.db.get_unprocessed_messages(sender, limit=50)
            if not messages:
                continue

            existing_likes = self.db.get_preferences(sender, PreferenceType.LIKE)
            existing_dislikes = self.db.get_preferences(sender, PreferenceType.DISLIKE)

            like_topics = [p.topic for p in existing_likes]
            dislike_topics = [p.topic for p in existing_dislikes]

            prompt_parts = [
                "Analyze these messages and find any topics the user likes or dislikes.",
                "Only extract clear preferences — things the user explicitly enjoys, "
                "is enthusiastic about, or expresses dislike for.",
                "Do NOT extract every noun — only genuine preferences.\n",
            ]

            if like_topics:
                prompt_parts.append(f"Already known likes: {', '.join(like_topics)}")
            if dislike_topics:
                prompt_parts.append(f"Already known dislikes: {', '.join(dislike_topics)}")
            if like_topics or dislike_topics:
                prompt_parts.append("Do NOT include topics already known above.\n")

            prompt_parts.append("Messages:")
            for msg in messages:
                prompt_parts.append(f'- "{msg.content}"')

            prompt = "\n".join(prompt_parts)

            try:
                response = await self._ollama_client.generate(
                    prompt=prompt,
                    tools=None,
                    format=MessagePreferences.model_json_schema(),
                )
                result = MessagePreferences.model_validate_json(response.content)

                for pref in result.preferences:
                    topic = pref.topic.lower().strip()
                    pref_type = pref.type.lower().strip()
                    if not topic or pref_type not in (PreferenceType.LIKE, PreferenceType.DISLIKE):
                        continue

                    opposite_type = (
                        PreferenceType.DISLIKE
                        if pref_type == PreferenceType.LIKE
                        else PreferenceType.LIKE
                    )
                    conflict = self.db.find_conflicting_preference(sender, topic, pref_type)

                    if conflict:
                        self.db.move_preference(sender, topic, opposite_type, pref_type)
                        await self._send_notification(
                            sender,
                            f"I moved {topic} from your {opposite_type}s to your {pref_type}s",
                        )
                        logger.info(
                            "Moved preference for %s: %s (%s → %s)",
                            sender,
                            topic,
                            opposite_type,
                            pref_type,
                        )
                        work_done = True
                    else:
                        existing = self.db.get_preferences(sender, pref_type)
                        if any(p.topic == topic for p in existing):
                            continue

                        self.db.add_preference(sender, topic, pref_type)
                        await self._send_notification(
                            sender,
                            f"I added {topic} to your {pref_type}s",
                        )
                        logger.info("Added %s preference for %s: %s", pref_type, sender, topic)
                        work_done = True

            except Exception as e:
                logger.error("Failed to analyze messages for %s: %s", sender, e)

            # Mark all messages as processed regardless of success
            self.db.mark_messages_processed([msg.id for msg in messages if msg.id is not None])

        return work_done

    async def _process_reaction(self, sender: str, reaction_msg: MessageLog) -> bool:
        """
        Process a single reaction and update preferences if needed.

        Args:
            sender: The user who reacted
            reaction_msg: The reaction message

        Returns:
            True if a preference was updated, False otherwise
        """
        emoji = reaction_msg.content
        if emoji in LIKE_REACTIONS:
            pref_type = PreferenceType.LIKE
        elif emoji in DISLIKE_REACTIONS:
            pref_type = PreferenceType.DISLIKE
        else:
            return False

        if not reaction_msg.parent_id:
            logger.debug("Reaction has no parent_id, skipping")
            return False

        reacted_to_msg = self.db.get_message_by_id(reaction_msg.parent_id)
        if not reacted_to_msg:
            logger.debug(
                "Could not find message for reaction parent_id: %s", reaction_msg.parent_id
            )
            return False

        try:
            prompt = (
                f"Extract the main topic or subject from this message. "
                f"Return a short phrase or keyword (1-4 words).\n\n"
                f'Message: "{reacted_to_msg.content}"'
            )

            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=ReactionAnalysis.model_json_schema(),
            )

            analysis = ReactionAnalysis.model_validate_json(response.content)
            topic = analysis.topic.lower().strip()

            if not topic:
                logger.debug("LLM returned empty topic for reaction")
                return False

            opposite_type = (
                PreferenceType.DISLIKE if pref_type == PreferenceType.LIKE else PreferenceType.LIKE
            )
            conflict = self.db.find_conflicting_preference(sender, topic, pref_type)

            if conflict:
                self.db.move_preference(sender, topic, opposite_type, pref_type)
                await self._send_notification(
                    sender,
                    f"I moved {topic} from your {opposite_type}s to your {pref_type}s",
                )
                logger.info(
                    "Moved preference for %s: %s (%s → %s)", sender, topic, opposite_type, pref_type
                )
                return True
            else:
                existing = self.db.get_preferences(sender, pref_type)
                if any(p.topic == topic for p in existing):
                    return False

                self.db.add_preference(sender, topic, pref_type)
                await self._send_notification(
                    sender,
                    f"I added {topic} to your {pref_type}s",
                )
                logger.info("Added %s preference for %s: %s", pref_type, sender, topic)
                return True

        except Exception as e:
            logger.error("Failed to process reaction: %s", e)
            return False

    async def _send_notification(self, recipient: str, message: str) -> None:
        """Send a notification message to the user."""
        if not self._channel:
            return

        typing_task = asyncio.create_task(self._channel._typing_loop(recipient))
        try:
            await self._channel.send_response(
                recipient,
                message,
                parent_id=None,
                attachments=None,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(recipient, False)
