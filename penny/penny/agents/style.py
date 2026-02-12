"""AdaptiveStyleAgent for learning user speaking styles."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from penny.agents.base import Agent

logger = logging.getLogger(__name__)

# Minimum messages required before style analysis begins
STYLE_MIN_MESSAGES = 20

# Number of recent messages to analyze for style profile
STYLE_ANALYSIS_WINDOW = 100

# System prompt for style analysis
STYLE_ANALYSIS_PROMPT = (
    """Analyze the following messages from a user and describe their """
    """unique communication style.

Focus on:
- Sentence length (short/punchy vs. long/detailed)
- Formality level (casual vs. formal)
- Emoji and punctuation habits (frequency, types)
- Capitalization patterns (all lowercase, proper case, Title Case, etc.)
- Common slang, abbreviations, or filler words (e.g., "ya", "tbh", "lol")

Respond with a natural language description (2-4 sentences) that can be used """
    """as part of a system prompt to help an AI agent match this style. Do not """
    """include direct instructions to the AI — just describe how the user writes.

Example output:
"User writes in short lowercase sentences with minimal punctuation. Uses slang """
    """like 'ya' and 'tbh' frequently. Rarely uses emojis."

Messages to analyze:"""
)


class StyleDescription(BaseModel):
    """Schema for LLM response: user communication style description."""

    description: str = Field(
        description="Natural language description of the user's communication style"
    )


class AdaptiveStyleAgent(Agent):
    """Agent for learning and adapting to user communication styles."""

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "style"

    async def execute(self) -> bool:
        """
        Analyze recent messages for each user and update their style profiles.

        Returns:
            True if work was done, False if nothing to do
        """
        senders = self.db.get_all_senders()
        if not senders:
            return False

        work_done = False
        for sender in senders:
            # Check if user has enough messages
            total_messages = self.db.count_user_messages(sender)
            if total_messages < STYLE_MIN_MESSAGES:
                continue

            # Check if we already analyzed today
            existing_profile = self.db.get_user_style_profile(sender)
            if existing_profile:
                now = datetime.now(UTC)
                hours_since_update = (now - existing_profile.updated_at).total_seconds() / 3600
                if hours_since_update < 24:
                    continue

            # Get last N user messages
            messages = self.db.get_user_messages_for_style(sender, limit=STYLE_ANALYSIS_WINDOW)
            if not messages:
                continue

            # Generate style profile
            style_prompt = await self._analyze_style(sender, messages)
            if style_prompt:
                # Upsert style profile
                self.db.upsert_user_style_profile(
                    user_id=sender,
                    style_prompt=style_prompt,
                    message_count=len(messages),
                )
                logger.info("Updated style profile for %s (%d messages)", sender, len(messages))
                work_done = True

        return work_done

    async def _analyze_style(self, sender: str, messages: list[str]) -> str | None:
        """
        Analyze messages and generate a style profile description.

        Args:
            sender: The user identifier
            messages: List of message content strings

        Returns:
            Style description string, or None if analysis failed
        """
        prompt_parts = [STYLE_ANALYSIS_PROMPT]
        for msg in messages:
            prompt_parts.append(f'- "{msg}"')

        prompt = "\n".join(prompt_parts)

        try:
            response = await self._ollama_client.generate(
                prompt=prompt,
                tools=None,
                format=StyleDescription.model_json_schema(),
            )
            result = StyleDescription.model_validate_json(response.content)
            return result.description.strip()

        except Exception as e:
            logger.error("Failed to analyze style for %s: %s", sender, e)
            return None
