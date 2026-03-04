"""HistoryAgent — daily conversation topic summarization and preference extraction.

Runs on a schedule. Each cycle, for each user:
1. Summarizes today's messages (midnight to now) via upsert — rolling update
2. Backfills completed past days that lack history entries
3. Extracts user preferences from sentiment and emoji reactions
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel
from pydantic import Field as PydanticField

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.ollama.embeddings import cosine_similarity, deserialize_embedding, serialize_embedding
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


class ExtractedPreference(BaseModel):
    """A single extracted preference."""

    content: str = PydanticField(description="The preference topic (3-10 words)")
    valence: str = PydanticField(description="'positive' or 'negative'")


class ExtractedPreferences(BaseModel):
    """Schema for preference extraction output."""

    preferences: list[ExtractedPreference] = PydanticField(
        default_factory=list,
        description="User preferences detected in the conversation",
    )


PREFERENCE_DEDUP_THRESHOLD = 0.85


class HistoryAgent(Agent):
    """Background worker that compacts daily conversations into topic summaries."""

    name = "history"

    def __init__(self, **kwargs: object) -> None:
        kwargs["system_prompt"] = Prompt.SUMMARIZE_TO_BULLETS
        super().__init__(**kwargs)  # type: ignore[arg-type]

    async def execute_for_user(self, user: str) -> bool:
        """Summarize today's conversation, extract preferences, then backfill past days."""
        did_work = await self._summarize_today(user)
        did_work = await self._extract_today_preferences(user) or did_work

        max_days = int(self.config.runtime.HISTORY_MAX_DAYS_PER_RUN)
        days = self._find_unsummarized_days(user, max_days)
        for day_start, day_end in days:
            await self._summarize_day(user, day_start, day_end)
            await self._extract_day_preferences(user, day_start, day_end)
            did_work = True

        return did_work

    # ── Topic summarization ───────────────────────────────────────────────

    async def _summarize_today(self, user: str) -> bool:
        """Summarize messages from midnight to now, upserting today's history entry."""
        day_start = self._midnight_today()
        day_end = datetime.now(UTC).replace(tzinfo=None)

        if self._already_rolled_up(user, day_start, day_end):
            return False

        messages = self.db.messages.get_messages_in_range(user, day_start, day_end)
        if not messages:
            return False

        message_text = self._format_messages(messages)
        response = await self.run(prompt=message_text)
        topics = response.answer.strip()
        if not topics:
            return False

        embedding = await self._embed_text(topics)
        self.db.history.upsert(
            user=user,
            period_start=day_start,
            period_end=day_end,
            duration=PennyConstants.HistoryDuration.DAILY,
            topics=topics,
            embedding=embedding,
        )
        logger.info("Today's history updated for %s", user)
        return True

    def _already_rolled_up(self, user: str, day_start: datetime, day_end: datetime) -> bool:
        """Check if the history entry is already up-to-date with the latest message."""
        existing = self.db.history.get_latest(user, PennyConstants.HistoryDuration.DAILY)
        if not existing or existing.period_start != day_start:
            return False
        latest_msg = self.db.messages.get_latest_message_time_in_range(user, day_start, day_end)
        if not latest_msg:
            return True
        return existing.created_at >= latest_msg

    async def _summarize_day(self, user: str, day_start: datetime, day_end: datetime) -> None:
        """Get messages for a day and call model to summarize topics."""
        messages = self.db.messages.get_messages_in_range(user, day_start, day_end)
        if not messages:
            logger.debug("No messages for %s on %s, skipping", user, day_start.date())
            return

        message_text = self._format_messages(messages)
        response = await self.run(prompt=message_text)
        topics = response.answer.strip()
        if not topics:
            logger.debug("Model returned empty topics for %s on %s", user, day_start.date())
            return

        embedding = await self._embed_text(topics)
        self.db.history.add(
            user=user,
            period_start=day_start,
            period_end=day_end,
            duration=PennyConstants.HistoryDuration.DAILY,
            topics=topics,
            embedding=embedding,
        )
        logger.info("History entry created for %s on %s", user, day_start.date())

    # ── Preference extraction ─────────────────────────────────────────────

    async def _extract_today_preferences(self, user: str) -> bool:
        """Extract preferences from today's messages (dedup via embedding)."""
        day_start = self._midnight_today()
        day_end = datetime.now(UTC).replace(tzinfo=None)
        return await self._extract_preferences(user, day_start, day_end, dedup=True)

    async def _extract_day_preferences(
        self, user: str, day_start: datetime, day_end: datetime
    ) -> None:
        """Extract preferences for a completed day (skip if already done)."""
        if self.db.preferences.exists_for_period(user, day_start):
            return
        await self._extract_preferences(user, day_start, day_end, dedup=False)

    async def _extract_preferences(
        self, user: str, start: datetime, end: datetime, dedup: bool
    ) -> bool:
        """Build preference prompt, call LLM, store results."""
        prompt = self._build_preference_prompt(user, start, end)
        if not prompt:
            return False

        raw_prefs = await self._call_preference_extraction(prompt)
        if not raw_prefs:
            return False

        valid_prefs = self._validate_preferences(raw_prefs)
        if not valid_prefs:
            return False

        await self._store_preferences(user, valid_prefs, start, end, dedup)
        return True

    def _build_preference_prompt(self, user: str, start: datetime, end: datetime) -> str | None:
        """Build the preference extraction prompt from messages and reactions."""
        messages = self.db.messages.get_messages_in_range(user, start, end)
        reactions = self.db.messages.get_reactions_in_range(user, start, end)
        if not messages and not reactions:
            return None

        parts: list[str] = []
        if messages:
            parts.append(self._format_messages(messages))
        if reactions:
            reaction_text = self._format_reactions(reactions)
            if reaction_text:
                parts.append(reaction_text)
        return "\n\n".join(parts) if parts else None

    def _format_reactions(self, reactions: list) -> str:
        """Format reactions with parent message context for the prompt."""
        lines: list[str] = []
        for reaction in reactions:
            ts = reaction.timestamp.strftime("%H:%M")
            valence = self._classify_reaction_emoji(reaction.content)
            if not valence:
                continue
            parent = self.db.messages.get_by_id(reaction.parent_id) if reaction.parent_id else None
            if parent:
                lines.append(f'[{ts}] User reacted {reaction.content} to: "{parent.content[:200]}"')
        return "Reactions:\n" + "\n".join(lines) if lines else ""

    @staticmethod
    def _classify_reaction_emoji(emoji: str) -> str | None:
        """Classify an emoji as positive, negative, or None (unknown)."""
        if emoji in PennyConstants.POSITIVE_REACTION_EMOJIS:
            return PennyConstants.PreferenceValence.POSITIVE
        if emoji in PennyConstants.NEGATIVE_REACTION_EMOJIS:
            return PennyConstants.PreferenceValence.NEGATIVE
        return None

    async def _call_preference_extraction(self, prompt: str) -> list[ExtractedPreference]:
        """Call LLM with structured output to extract preferences."""
        full_prompt = f"{Prompt.PREFERENCE_EXTRACTION_PROMPT}\n\n{prompt}"
        try:
            response = await self._model_client.generate(
                prompt=full_prompt,
                tools=None,
                format=ExtractedPreferences.model_json_schema(),
            )
            if not response.content or not response.content.strip():
                return []
            result = ExtractedPreferences.model_validate_json(response.content)
            return result.preferences
        except Exception as e:
            logger.error("Preference extraction failed: %s", e)
            return []

    @staticmethod
    def _validate_preferences(
        prefs: list[ExtractedPreference],
    ) -> list[ExtractedPreference]:
        """Filter out preferences with invalid valence or empty content."""
        valid_valences = {
            PennyConstants.PreferenceValence.POSITIVE,
            PennyConstants.PreferenceValence.NEGATIVE,
        }
        return [p for p in prefs if p.valence in valid_valences and p.content.strip()]

    async def _store_preferences(
        self,
        user: str,
        prefs: list[ExtractedPreference],
        start: datetime,
        end: datetime,
        dedup: bool,
    ) -> None:
        """Compute embeddings and store preferences, optionally deduplicating."""
        for pref in prefs:
            embedding = await self._embed_text(pref.content)

            if dedup and embedding and self._is_duplicate_preference(user, embedding):
                logger.debug("Skipping duplicate preference: %s", pref.content[:50])
                continue

            self.db.preferences.add(
                user=user,
                content=pref.content.strip(),
                valence=pref.valence,
                source_period_start=start,
                source_period_end=end,
                embedding=embedding,
            )
            logger.info("Preference stored for %s: %s (%s)", user, pref.content[:50], pref.valence)

    def _is_duplicate_preference(self, user: str, embedding: bytes) -> bool:
        """Check if a preference with similar embedding already exists."""
        existing = self.db.preferences.get_with_embeddings(user)
        if not existing:
            return False
        candidate_vec = deserialize_embedding(embedding)
        for pref in existing:
            if pref.embedding:
                existing_vec = deserialize_embedding(pref.embedding)
                if cosine_similarity(candidate_vec, existing_vec) >= PREFERENCE_DEDUP_THRESHOLD:
                    return True
        return False

    # ── Shared helpers ────────────────────────────────────────────────────

    async def _embed_text(self, text: str) -> bytes | None:
        """Compute and serialize embedding for a text string."""
        if not self._embedding_model_client:
            return None
        try:
            vecs = await self._embedding_model_client.embed(text)
            return serialize_embedding(vecs[0])
        except Exception as e:
            logger.warning("Failed to embed text: %s", e)
            return None

    def _find_unsummarized_days(self, user: str, max_days: int) -> list[tuple[datetime, datetime]]:
        """Find completed calendar days (UTC) without history entries."""
        duration = PennyConstants.HistoryDuration.DAILY
        latest = self.db.history.get_latest(user, duration)
        start = self._resolve_start_date(user, latest)
        if start is None:
            return []

        yesterday_end = self._midnight_today()
        days: list[tuple[datetime, datetime]] = []
        cursor = start
        while cursor < yesterday_end and len(days) < max_days:
            day_end = cursor + timedelta(days=1)
            if not self.db.history.exists(user, cursor, duration):
                days.append((cursor, day_end))
            cursor = day_end

        return days

    def _resolve_start_date(self, user: str, latest: object | None) -> datetime | None:
        """Determine where to start scanning for un-rolled-up days."""
        if latest is not None:
            return getattr(latest, "period_end", None)
        first_msg_time = self.db.messages.get_first_message_time(user)
        if first_msg_time is None:
            return None
        return first_msg_time.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _midnight_today() -> datetime:
        """Return midnight UTC for today as a naive datetime.

        Naive because SQLite strips timezone info — all stored datetimes are naive UTC.
        """
        return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    @staticmethod
    def _format_messages(messages: list) -> str:
        """Format messages for the summarization prompt."""
        lines: list[str] = []
        for msg in messages:
            ts = msg.timestamp.strftime("%H:%M")
            if msg.direction == PennyConstants.MessageDirection.INCOMING:
                lines.append(f"[{ts}] User: {msg.content}")
            else:
                lines.append(f"[{ts}] Penny: {msg.content}")
        return "\n".join(lines)
