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
from penny.database.models import Preference
from penny.ollama.embeddings import deserialize_embedding, serialize_embedding
from penny.ollama.similarity import DedupStrategy, is_embedding_duplicate
from penny.prompts import Prompt

logger = logging.getLogger(__name__)


class IdentifiedPreferenceTopics(BaseModel):
    """Schema for pass 1: preference topics found in conversation."""

    new: list[str] = PydanticField(
        default_factory=list,
        description="New preference topics not already known (3-10 words each)",
    )
    existing: list[str] = PydanticField(
        default_factory=list,
        description="Already-known preference content strings discussed in the conversation",
    )


class ClassifiedPreference(BaseModel):
    """A preference topic with its valence classification."""

    content: str = PydanticField(description="The preference topic")
    valence: str = PydanticField(description="'positive' or 'negative'")


class ClassifiedPreferences(BaseModel):
    """Schema for pass 2: valence classification of preference topics."""

    preferences: list[ClassifiedPreference] = PydanticField(
        default_factory=list,
        description="Preference topics with valence classifications",
    )


class HistoryAgent(Agent):
    """Background worker that compacts daily conversations into topic summaries."""

    name = "history"

    def __init__(self, **kwargs: object) -> None:
        kwargs["system_prompt"] = Prompt.SUMMARIZE_TO_BULLETS
        super().__init__(**kwargs)  # type: ignore[arg-type]

    async def execute_for_user(self, user: str) -> bool:
        """Summarize today's conversation, extract preferences, backfill days and weeks."""
        did_work = await self._summarize_today(user)
        did_work = await self._extract_today_preferences(user) or did_work

        max_days = int(self.config.runtime.HISTORY_MAX_DAYS_PER_RUN)
        days = self._find_unsummarized_days(user, max_days)
        for day_start, day_end in days:
            await self._summarize_day(user, day_start, day_end)
            await self._extract_day_preferences(user, day_start, day_end)
            did_work = True

        did_work = await self._rollup_completed_weeks(user) or did_work

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

    # ── Weekly rollup ──────────────────────────────────────────────────────

    async def _rollup_completed_weeks(self, user: str) -> bool:
        """Summarize completed weeks from daily entries into weekly history entries."""
        max_weeks = 2
        weeks = self._find_unrolled_weeks(user, max_weeks)
        if not weeks:
            return False

        did_work = False
        for week_start, week_end in weeks:
            input_text = self._gather_weekly_input(user, week_start, week_end)
            if not input_text:
                continue

            response = await self.run(prompt=input_text)
            topics = response.answer.strip()
            if not topics:
                continue

            embedding = await self._embed_text(topics)
            self.db.history.add(
                user=user,
                period_start=week_start,
                period_end=week_end,
                duration=PennyConstants.HistoryDuration.WEEKLY,
                topics=topics,
                embedding=embedding,
            )
            logger.info("Weekly rollup created for %s: %s", user, week_start.date())
            did_work = True
        return did_work

    def _find_unrolled_weeks(self, user: str, max_weeks: int) -> list[tuple[datetime, datetime]]:
        """Find completed ISO weeks with daily entries but no weekly entry."""
        daily = PennyConstants.HistoryDuration.DAILY
        weekly = PennyConstants.HistoryDuration.WEEKLY

        earliest_daily = self.db.history.get_recent(user, daily, limit=1)
        if not earliest_daily:
            return []

        first_start = earliest_daily[0].period_start
        first_monday = first_start - timedelta(days=first_start.weekday())
        first_monday = first_monday.replace(hour=0, minute=0, second=0, microsecond=0)

        today = self._midnight_today()
        current_monday = today - timedelta(days=today.weekday())

        weeks: list[tuple[datetime, datetime]] = []
        cursor = first_monday
        while cursor < current_monday and len(weeks) < max_weeks:
            week_end = cursor + timedelta(days=7)
            has_daily = bool(self.db.history.get_in_range(user, daily, cursor, week_end))
            has_weekly = self.db.history.exists(user, cursor, weekly)
            if has_daily and not has_weekly:
                weeks.append((cursor, week_end))
            cursor = week_end
        return weeks

    def _gather_weekly_input(self, user: str, week_start: datetime, week_end: datetime) -> str:
        """Concatenate daily topic entries for a week as input text."""
        daily = PennyConstants.HistoryDuration.DAILY
        entries = self.db.history.get_in_range(user, daily, week_start, week_end)
        if not entries:
            return ""

        lines: list[str] = []
        for entry in entries:
            date_label = entry.period_start.strftime("%b %-d")
            lines.append(f"{date_label}:")
            lines.append(entry.topics.strip())
        return "\n".join(lines)

    # ── Preference extraction ─────────────────────────────────────────────

    async def _extract_today_preferences(self, user: str) -> bool:
        """Extract preferences from unprocessed messages only."""
        messages = self.db.messages.get_unprocessed(user, limit=100)
        reactions = self.db.messages.get_user_reactions(user, limit=100)
        if not messages and not reactions:
            return False

        conversation = self._build_unprocessed_content(messages, reactions)
        if not conversation:
            return False

        now = datetime.now(UTC).replace(tzinfo=None)
        did_work = await self._extract_preferences_from_content(
            user, conversation, self._midnight_today(), now
        )
        if did_work:
            message_ids = [m.id for m in messages if m.id is not None]
            reaction_ids = [r.id for r in reactions if r.id is not None]
            self.db.messages.mark_processed(message_ids + reaction_ids)
        return did_work

    async def _extract_day_preferences(
        self, user: str, day_start: datetime, day_end: datetime
    ) -> None:
        """Extract preferences for a completed day (skip if already done)."""
        if self.db.preferences.exists_for_period(user, day_start):
            return
        conversation = self._build_conversation_content(user, day_start, day_end)
        if conversation:
            await self._extract_preferences_from_content(user, conversation, day_start, day_end)

    async def _extract_preferences_from_content(
        self, user: str, conversation: str, start: datetime, end: datetime
    ) -> bool:
        """Two-pass preference extraction: identify topics, dedup, classify valence."""
        existing = self.db.preferences.get_for_user(user)

        identified = await self._identify_preference_topics(conversation, existing)
        if not identified:
            return False

        bumped_ids = self._bump_existing_mentions(identified.existing, existing)

        new_topics = self._filter_new_topics(identified, existing)
        if not new_topics:
            return bool(identified.existing)

        survivors = await self._dedup_preference_topics(new_topics, existing, bumped_ids)
        if not survivors:
            return bool(identified.existing)

        survivor_topics = [topic for topic, _emb in survivors]
        classified = await self._classify_preference_valence(survivor_topics, conversation)
        if not classified:
            return bool(identified.existing)

        embedding_lookup = {topic.lower(): emb for topic, emb in survivors}
        self._store_classified_preferences(user, classified, start, end, embedding_lookup)
        return True

    def _bump_existing_mentions(self, mentioned: list[str], existing: list[Preference]) -> set[int]:
        """Increment mention_count for existing preferences the LLM recognized.

        Returns set of preference IDs that were bumped (to avoid double-counting in dedup).
        """
        bumped: set[int] = set()
        content_to_pref = {p.content.lower(): p for p in existing}
        for name in mentioned:
            pref = content_to_pref.get(name.strip().lower())
            if pref and pref.id is not None:
                self.db.preferences.increment_mention_count(pref.id)
                bumped.add(pref.id)
                logger.info("Preference '%s' mention count incremented", pref.content[:50])
        return bumped

    @staticmethod
    def _filter_new_topics(
        identified: IdentifiedPreferenceTopics, existing: list[Preference]
    ) -> list[str]:
        """Filter new topics to exclude anything already in existing or known preferences."""
        existing_lower = {p.content.lower() for p in existing}
        identified_existing_lower = {n.strip().lower() for n in identified.existing}
        exclude = existing_lower | identified_existing_lower
        return [t.strip() for t in identified.new if t.strip() and t.strip().lower() not in exclude]

    # ── Pass 1: topic identification ─────────────────────────────────────

    def _build_unprocessed_content(self, messages: list, reactions: list) -> str | None:
        """Build conversation text from pre-fetched unprocessed messages and reactions."""
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

    def _build_conversation_content(self, user: str, start: datetime, end: datetime) -> str | None:
        """Build user-only conversation text for preference extraction (backfill path).

        Only includes incoming (user) messages and reactions — Penny's
        responses are excluded so the model doesn't extract Penny's
        topics as user preferences.
        """
        messages = self.db.messages.get_messages_in_range(user, start, end)
        reactions = self.db.messages.get_reactions_in_range(user, start, end)
        user_messages = [
            m for m in messages if m.direction == PennyConstants.MessageDirection.INCOMING
        ]
        if not user_messages and not reactions:
            return None

        parts: list[str] = []
        if user_messages:
            parts.append(self._format_messages(user_messages))
        if reactions:
            reaction_text = self._format_reactions(reactions)
            if reaction_text:
                parts.append(reaction_text)
        return "\n\n".join(parts) if parts else None

    async def _identify_preference_topics(
        self, conversation: str, existing: list
    ) -> IdentifiedPreferenceTopics | None:
        """Pass 1: ask model to identify new and existing preference topics."""
        known_context = self._build_known_preferences_context(existing)
        prompt = f"{Prompt.PREFERENCE_IDENTIFICATION_PROMPT}\n\n{conversation}"
        if known_context:
            prompt += f"\n\n{known_context}"

        try:
            response = await self._model_client.generate(
                prompt=prompt,
                tools=None,
                format=IdentifiedPreferenceTopics.model_json_schema(),
            )
            if not response.content or not response.content.strip():
                return None
            return IdentifiedPreferenceTopics.model_validate_json(response.content)
        except Exception as e:
            logger.error("Preference topic identification failed: %s", e)
            return None

    @staticmethod
    def _build_known_preferences_context(existing: list) -> str:
        """Format existing preferences so the model can skip already-known topics."""
        if not existing:
            return ""
        lines: list[str] = []
        for pref in existing:
            lines.append(f"- {pref.valence}: {pref.content}")
        return (
            "Already known preferences (do NOT re-extract these or rephrasings of these):\n"
            + "\n".join(lines)
        )

    # ── Dedup ────────────────────────────────────────────────────────────

    async def _dedup_preference_topics(
        self,
        topics: list[str],
        existing_prefs: list[Preference],
        already_bumped: set[int] | None = None,
    ) -> list[tuple[str, bytes | None]]:
        """Embed topics, dedup against existing, increment mention count on match.

        When a topic matches an existing DB preference, increments its mention_count
        instead of silently skipping (unless already bumped this pass).
        New unique topics are returned as survivors.
        """
        existing_items: list[tuple[str, bytes | None]] = [
            (p.content, p.embedding) for p in existing_prefs
        ]
        db_pref_count = len(existing_prefs)
        bumped = already_bumped or set()
        survivors: list[tuple[str, bytes | None]] = []

        for topic in topics:
            embedding = await self._embed_text(topic)
            candidate_vec = deserialize_embedding(embedding) if embedding else None

            match_idx = is_embedding_duplicate(
                topic,
                candidate_vec,
                existing_items,
                DedupStrategy.TCR_OR_EMBEDDING,
                embedding_threshold=self.config.runtime.PREFERENCE_DEDUP_EMBEDDING_THRESHOLD,
                tcr_threshold=self.config.runtime.PREFERENCE_DEDUP_TCR_THRESHOLD,
            )
            if match_idx is not None:
                self._handle_dedup_match(match_idx, db_pref_count, existing_prefs, topic, bumped)
                continue

            survivors.append((topic, embedding))
            existing_items.append((topic, embedding))
        return survivors

    def _handle_dedup_match(
        self,
        match_idx: int,
        db_pref_count: int,
        existing_prefs: list[Preference],
        topic: str,
        already_bumped: set[int],
    ) -> None:
        """Increment mention count for DB matches, skip if already bumped this pass."""
        if match_idx < db_pref_count:
            matched = existing_prefs[match_idx]
            if matched.id in already_bumped:
                logger.debug("Skipping already-bumped preference: '%s'", matched.content[:50])
                return
            self.db.preferences.increment_mention_count(matched.id)  # type: ignore[arg-type]
            already_bumped.add(matched.id)  # type: ignore[arg-type]
            logger.info(
                "Preference '%s' mention count incremented (matches '%s')",
                topic[:50],
                matched.content[:50],
            )
        else:
            logger.debug("Skipping intra-batch duplicate: '%s'", topic[:50])

    # ── Pass 2: valence classification ───────────────────────────────────

    async def _classify_preference_valence(
        self, topics: list[str], conversation: str
    ) -> list[ClassifiedPreference]:
        """Pass 2: classify each topic as positive or negative."""
        topics_text = "\n".join(f"- {t}" for t in topics)
        prompt = (
            f"{Prompt.PREFERENCE_VALENCE_PROMPT}\n\n"
            f"Topics to classify:\n{topics_text}\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            response = await self._model_client.generate(
                prompt=prompt,
                tools=None,
                format=ClassifiedPreferences.model_json_schema(),
            )
            if not response.content or not response.content.strip():
                return []
            result = ClassifiedPreferences.model_validate_json(response.content)
            valid_valences = {
                PennyConstants.PreferenceValence.POSITIVE,
                PennyConstants.PreferenceValence.NEGATIVE,
            }
            return [
                p for p in result.preferences if p.valence in valid_valences and p.content.strip()
            ]
        except Exception as e:
            logger.error("Preference valence classification failed: %s", e)
            return []

    # ── Storage ──────────────────────────────────────────────────────────

    def _store_classified_preferences(
        self,
        user: str,
        classified: list[ClassifiedPreference],
        start: datetime,
        end: datetime,
        embedding_lookup: dict[str, bytes | None],
    ) -> None:
        """Store classified preferences with pre-computed embeddings."""
        for pref in classified:
            content = pref.content.strip()
            embedding = embedding_lookup.get(content.lower())
            self.db.preferences.add(
                user=user,
                content=content,
                valence=pref.valence,
                source_period_start=start,
                source_period_end=end,
                embedding=embedding,
                source=PennyConstants.PreferenceSource.EXTRACTED,
            )
            logger.info("Preference stored for %s: %s (%s)", user, content[:50], pref.valence)

    # ── Reaction helpers ─────────────────────────────────────────────────

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
