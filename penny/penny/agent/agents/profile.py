"""ProfileAgent for building user profiles from message history."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from penny.agent.base import Agent
from penny.constants import DISLIKE_REACTIONS, LIKE_REACTIONS, PROFILE_PROMPT, PreferenceType

if TYPE_CHECKING:
    from penny.channels import MessageChannel
    from penny.database.models import MessageLog

logger = logging.getLogger(__name__)


class ReactionAnalysis(BaseModel):
    """Schema for parsing reaction analysis from LLM."""

    topic: str = Field(
        description="The main topic or subject of the message (a short phrase or keyword)"
    )


class ProfileAgent(Agent):
    """Agent for generating user profiles from message history and analyzing reactions."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "profile"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending notifications."""
        self._channel = channel

    async def execute(self) -> bool:
        """
        Process profile updates and reaction analysis.

        Returns:
            True if work was done, False if nothing to do
        """
        # First, analyze reactions for all users
        reaction_work = await self._analyze_reactions()

        # Then, generate topics for users needing profile updates
        profile_work = await self._generate_topics()

        return reaction_work or profile_work

    async def _analyze_reactions(self) -> bool:
        """
        Analyze recent reactions and update preferences.

        Returns:
            True if reactions were processed, False if nothing to do
        """
        if not self._channel:
            logger.warning("ProfileAgent: no channel set, skipping reaction analysis")
            return False

        # Get all users who have sent messages
        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False
        for sender in senders:
            # Get recent reactions from this user
            reactions = self.db.get_user_reactions(sender, limit=10)
            if not reactions:
                continue

            # Process each reaction
            for reaction_msg in reactions:
                if await self._process_reaction(sender, reaction_msg):
                    work_done = True

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
        # Determine sentiment from emoji
        emoji = reaction_msg.content
        if emoji in LIKE_REACTIONS:
            pref_type = PreferenceType.LIKE
        elif emoji in DISLIKE_REACTIONS:
            pref_type = PreferenceType.DISLIKE
        else:
            # Unknown reaction type, skip
            return False

        # Find the message that was reacted to
        if not reaction_msg.external_id:
            logger.debug("Reaction has no external_id, skipping")
            return False

        reacted_to_msg = self.db.find_message_by_external_id(reaction_msg.external_id)
        if not reacted_to_msg:
            logger.debug(
                "Could not find message for reaction external_id: %s", reaction_msg.external_id
            )
            return False

        # Use LLM to extract topic from the reacted message
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

            # Check for conflicts and update preferences
            opposite_type = (
                PreferenceType.DISLIKE if pref_type == PreferenceType.LIKE else PreferenceType.LIKE
            )
            conflict = self.db.find_conflicting_preference(sender, topic, pref_type)

            if conflict:
                # Move from opposite list to new list
                self.db.move_preference(sender, topic, opposite_type, pref_type)
                await self._send_notification(
                    sender,
                    f"i moved {topic} from your {opposite_type}s to your {pref_type}s",
                )
                logger.info(
                    "Moved preference for %s: %s (%s → %s)", sender, topic, opposite_type, pref_type
                )
                return True
            else:
                # Check if already in the correct list
                existing = self.db.get_preferences(sender, pref_type)
                if any(p.topic == topic for p in existing):
                    # Already tracked, no action needed
                    return False

                # Add to new list
                self.db.add_preference(sender, topic, pref_type)
                await self._send_notification(
                    sender,
                    f"i added {topic} to your {pref_type}s",
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

    async def _generate_topics(self) -> bool:
        """
        Generate topics for users needing profile updates.

        Returns:
            True if a profile was generated/updated, False if nothing to do
        """
        users = self.db.get_users_needing_profile_update()
        if not users:
            return False

        # Process one user per execution (like SummarizeAgent)
        sender = users[0]
        messages = self.db.get_user_messages(sender)

        if not messages:
            return False

        logger.info("Generating profile for user %s (%d messages)", sender, len(messages))

        # Format messages for the prompt
        messages_text = self._format_messages(messages)
        prompt = f"{PROFILE_PROMPT}{messages_text}"

        response = await self.run(prompt=prompt)
        profile_text = response.answer.strip() if response.answer else None

        if profile_text:
            # Use the timestamp of the newest message
            last_timestamp = messages[-1].timestamp
            self.db.save_user_topics(sender, profile_text, last_timestamp)
            logger.info("Generated topics for %s (length: %d)", sender, len(profile_text))
            return True

        return False

    def _format_messages(self, messages: list[MessageLog]) -> str:
        """Format user messages for profile generation."""
        return "\n".join(f"[{m.timestamp.strftime('%Y-%m-%d')}] {m.content}" for m in messages)
