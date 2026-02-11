"""PreferenceAgent for extracting user preferences from messages and reactions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import (
    DISLIKE_REACTIONS,
    LIKE_REACTIONS,
    PREFERENCE_BATCH_LIMIT,
    PreferenceType,
)

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class ExtractedTopics(BaseModel):
    """Schema for LLM response: list of newly discovered preference topics."""

    topics: list[str] = Field(
        default_factory=list,
        description="List of new topics found (short phrases, 1-4 words each)",
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
        Extract preferences from reactions and messages.

        Runs two passes per user: one for likes, one for dislikes.
        Each pass combines reacted-to message content with regular user messages,
        passes existing preferences as context, and asks the LLM for new topics.

        Returns:
            True if work was done, False if nothing to do
        """
        if not self._channel:
            logger.warning("PreferenceAgent: no channel set, skipping")
            return False

        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False
        for sender in senders:
            reactions = self.db.get_user_reactions(sender, limit=PREFERENCE_BATCH_LIMIT)
            messages = self.db.get_unprocessed_messages(sender, limit=PREFERENCE_BATCH_LIMIT)

            if not reactions and not messages:
                continue

            # Resolve like/dislike reactions to their parent message content
            like_reaction_texts: list[str] = []
            dislike_reaction_texts: list[str] = []
            for reaction in reactions:
                emoji = reaction.content
                if emoji not in LIKE_REACTIONS and emoji not in DISLIKE_REACTIONS:
                    continue
                if not reaction.parent_id:
                    continue
                parent_msg = self.db.get_message_by_id(reaction.parent_id)
                if not parent_msg:
                    continue
                if emoji in LIKE_REACTIONS:
                    like_reaction_texts.append(parent_msg.content)
                else:
                    dislike_reaction_texts.append(parent_msg.content)

            user_message_texts = [msg.content for msg in messages]

            # Two passes: likes then dislikes
            for pref_type, reaction_texts in [
                (PreferenceType.LIKE, like_reaction_texts),
                (PreferenceType.DISLIKE, dislike_reaction_texts),
            ]:
                if not reaction_texts and not user_message_texts:
                    continue

                updated = await self._extract_and_store(
                    sender, pref_type, reaction_texts, user_message_texts
                )
                if updated:
                    work_done = True

            # Mark all reactions and messages as processed
            reaction_ids = [r.id for r in reactions if r.id is not None]
            message_ids = [m.id for m in messages if m.id is not None]
            if reaction_ids:
                self.db.mark_messages_processed(reaction_ids)
            if message_ids:
                self.db.mark_messages_processed(message_ids)

        return work_done

    async def _extract_and_store(
        self,
        sender: str,
        pref_type: str,
        reaction_texts: list[str],
        user_message_texts: list[str],
    ) -> bool:
        """
        Run a single LLM pass for one preference type (like or dislike).

        Args:
            sender: The user identifier
            pref_type: "like" or "dislike"
            reaction_texts: Content of messages the user reacted to with matching emoji
            user_message_texts: Content of unprocessed regular user messages

        Returns:
            True if any new preferences were added
        """
        existing = self.db.get_preferences(sender, pref_type)
        existing_topics = [p.topic for p in existing]

        sentiment_desc = (
            "enjoys or is enthusiastic about"
            if pref_type == PreferenceType.LIKE
            else "dislikes or expresses negativity toward"
        )

        prompt_parts = [
            f"Find any NEW topics the user {pref_type}s from the messages below.",
            f"Only extract clear {pref_type}s — things the user explicitly {sentiment_desc}.",
            "Do NOT extract every noun — only genuine preferences.",
            "Return short phrases (1-4 words each).\n",
        ]

        if existing_topics:
            prompt_parts.append(f"Already known {pref_type}s: {', '.join(existing_topics)}")
            prompt_parts.append("Do NOT include topics already known above.\n")

        if reaction_texts:
            prompt_parts.append(f"Messages the user reacted to with a {pref_type} emoji:")
            for text in reaction_texts:
                prompt_parts.append(f'- "{text}"')
            prompt_parts.append("")

        if user_message_texts:
            prompt_parts.append("Messages from the user:")
            for text in user_message_texts:
                prompt_parts.append(f'- "{text}"')

        prompt = "\n".join(prompt_parts)

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=ExtractedTopics.model_json_schema(),
            )
            result = ExtractedTopics.model_validate_json(response.content)

            added_topics: list[str] = []
            for raw_topic in result.topics:
                topic = raw_topic.lower().strip()
                if not topic:
                    continue

                # Skip if already known
                if any(p.topic == topic for p in existing):
                    continue

                self.db.add_preference(sender, topic, pref_type)
                logger.info("Added %s preference for %s: %s", pref_type, sender, topic)
                added_topics.append(topic)

            # Send a single batched notification for all new preferences
            if added_topics:
                if len(added_topics) == 1:
                    message = f"I added {added_topics[0]} to your {pref_type}s"
                else:
                    bullet_list = "\n".join(f"• {topic}" for topic in added_topics)
                    message = f"I added these to your {pref_type}s:\n{bullet_list}"
                await self._send_notification(sender, message)

            return len(added_topics) > 0

        except Exception as e:
            logger.error("Failed to extract %s preferences for %s: %s", pref_type, sender, e)
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
