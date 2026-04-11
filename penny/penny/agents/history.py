"""HistoryAgent — preference extraction and knowledge extraction.

Runs on a schedule. Each cycle:
1. Extracts knowledge from browse tool results (user-independent)
2. Per user: extracts preferences from unprocessed text messages
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel
from pydantic import Field as PydanticField
from similarity.dedup import DedupStrategy, is_embedding_duplicate

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import Preference, PromptLog
from penny.llm.embeddings import deserialize_embedding, serialize_embedding
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


class HistoryPromptType:
    """Prompt types for HistoryAgent flows."""

    PREFERENCE_IDENTIFICATION = "preference_identification"
    PREFERENCE_VALENCE = "preference_valence"
    KNOWLEDGE_SUMMARIZE = "knowledge_summarize"


class HistoryAgent(Agent):
    """Background worker that extracts knowledge and user preferences."""

    name = "history"

    async def execute(self) -> bool:
        """Extract knowledge (user-independent), then run per-user work."""
        knowledge_work = await self._extract_knowledge()
        user_work = await super().execute()
        return knowledge_work or user_work

    async def execute_for_user(self, user: str) -> bool:
        """Extract preferences from unprocessed messages."""
        run_id = uuid.uuid4().hex
        return await self._extract_today_preferences(user, run_id)

    # ── Knowledge extraction ──────────────────────────────────────────────

    async def _extract_knowledge(self) -> bool:
        """Scan prompt logs for browse results and summarize into knowledge entries.

        Within a batch the same URL often appears in many prompts because each
        step of an agentic loop re-logs the prior tool result messages. Dedup by
        URL keeping the latest occurrence so each page is summarized at most once
        per batch instead of re-aggregating identical content N times.
        """
        watermark = self.db.knowledge.get_latest_prompt_timestamp() or datetime.min
        batch_limit = int(self.config.runtime.KNOWLEDGE_EXTRACTION_BATCH_LIMIT)
        prompts = self.db.messages.get_prompts_with_browse_after(watermark, batch_limit)
        if not prompts:
            return False

        unique_by_url = self._dedup_browse_results_by_url(prompts)
        if not unique_by_url:
            return False

        run_id = uuid.uuid4().hex
        for url, (title, content, prompt_id) in unique_by_url.items():
            await self._summarize_knowledge(url, title, content, prompt_id, run_id)
        return True

    @staticmethod
    def _dedup_browse_results_by_url(
        prompts: list[PromptLog],
    ) -> dict[str, tuple[str, str, int]]:
        """Collapse browse results across the batch to one entry per URL.

        URLs are normalized (fragment stripped, host lowercased) before keying
        so `/page` and `/page#anchor` collapse to a single entry. Iterates
        prompts in order; later occurrences overwrite earlier ones so the
        freshest content for each URL wins. Returns {url: (title, content,
        prompt_id)} keyed by the normalized URL.
        """
        unique: dict[str, tuple[str, str, int]] = {}
        for prompt in prompts:
            if prompt.id is None:
                continue
            for url, title, content in HistoryAgent._parse_browse_results(prompt):
                unique[HistoryAgent._normalize_url(url)] = (title, content, prompt.id)
        return unique

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Canonicalize a URL for dedup and storage.

        Strips the `#fragment` (client-side anchor, never affects page content)
        and lowercases the scheme and host (case-insensitive per RFC 3986).
        Path, query, and userinfo are preserved as-is — they can be
        case-sensitive on the server side. URLs that fail to parse are
        returned unchanged so a malformed string still keys consistently.
        """
        try:
            parts = urlsplit(url)
        except ValueError:
            return url
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))

    @staticmethod
    def _parse_browse_results(prompt: PromptLog) -> list[tuple[str, str, str]]:
        """Extract (url, title, page_content) tuples from browse tool results."""
        results: list[tuple[str, str, str]] = []
        for message in prompt.get_messages():
            if message.get("role") != "tool":
                continue
            content = message.get("content", "")
            for section in content.split(PennyConstants.SECTION_SEPARATOR):
                if section.startswith(PennyConstants.BROWSE_PAGE_HEADER):
                    parsed = HistoryAgent._parse_browse_section(section)
                    if parsed:
                        results.append(parsed)
        return results

    @staticmethod
    def _parse_browse_section(section: str) -> tuple[str, str, str] | None:
        """Parse a successful browse section into (url, title, page_content).

        Only matches the healthy format: header + Title: + URL: + content.
        Error responses (disconnects, timeouts, blocked domains) are skipped.
        """
        lines = section.split("\n", 3)
        if len(lines) < 3:
            return None
        url = lines[0][len(PennyConstants.BROWSE_PAGE_HEADER) :].strip()
        if not url:
            return None
        if not lines[1].startswith(PennyConstants.BROWSE_TITLE_PREFIX):
            return None
        if not lines[2].startswith(PennyConstants.BROWSE_URL_PREFIX):
            return None
        title = lines[1][len(PennyConstants.BROWSE_TITLE_PREFIX) :]
        page_content = lines[3] if len(lines) > 3 else ""
        if not page_content.strip():
            return None
        return (url, title, page_content)

    async def _summarize_knowledge(
        self, url: str, title: str, content: str, prompt_id: int, run_id: str
    ) -> None:
        """Summarize page content and upsert into knowledge store."""
        existing = self.db.knowledge.get_by_url(url)
        if existing:
            summary = await self._aggregate_knowledge(existing.summary, content, run_id)
        else:
            summary = await self._summarize_page(content, run_id)
        if not summary:
            return
        embedding = await self._embed_text(summary)
        self.db.knowledge.upsert_by_url(url, title, summary, embedding, prompt_id)

    async def _summarize_page(self, content: str, run_id: str) -> str | None:
        """Summarize a single page via LLM."""
        messages = [
            {"role": "system", "content": Prompt.KNOWLEDGE_SUMMARIZE},
            {"role": "user", "content": content},
        ]
        response = await self._model_client.chat(
            messages,
            agent_name=self.name,
            prompt_type=HistoryPromptType.KNOWLEDGE_SUMMARIZE,
            run_id=run_id,
        )
        return response.content.strip() if response.content else None

    async def _aggregate_knowledge(
        self, existing_summary: str, new_content: str, run_id: str
    ) -> str | None:
        """Merge existing summary with new page content via LLM."""
        user_content = f"Existing summary:\n{existing_summary}\n\nNew content:\n{new_content}"
        messages = [
            {"role": "system", "content": Prompt.KNOWLEDGE_AGGREGATE},
            {"role": "user", "content": user_content},
        ]
        response = await self._model_client.chat(
            messages,
            agent_name=self.name,
            prompt_type=HistoryPromptType.KNOWLEDGE_SUMMARIZE,
            run_id=run_id,
        )
        return response.content.strip() if response.content else None

    # ── Preference extraction ─────────────────────────────────────────────

    async def _extract_today_preferences(self, user: str, run_id: str) -> bool:
        """Extract preferences from unprocessed text messages only.

        Reactions are processed for thought valence only (no preference extraction)
        and are always marked processed so they don't accumulate.
        """
        messages = self.db.messages.get_unprocessed(user, limit=100)
        reactions = self.db.messages.get_user_reactions(user, limit=100)
        if not messages and not reactions:
            return False

        processed_ids: list[int] = []
        did_work = False

        if messages:
            did_work = await self._extract_text_preferences(user, messages, run_id)
            if did_work:
                processed_ids.extend(m.id for m in messages if m.id is not None)

        if reactions:
            self._process_reactions(reactions)
            processed_ids.extend(r.id for r in reactions if r.id is not None)

        if processed_ids:
            self.db.messages.mark_processed(processed_ids)
        return did_work

    async def _extract_text_preferences(self, user: str, messages: list, run_id: str) -> bool:
        """Extract preferences from text messages via two-pass LLM pipeline."""
        conversation = self._format_messages(messages)
        if not conversation:
            return False
        return await self._extract_preferences_from_content(user, conversation, run_id)

    async def _extract_preferences_from_content(
        self, user: str, conversation: str, run_id: str
    ) -> bool:
        """Two-pass preference extraction: identify topics, dedup, classify valence."""
        existing = self.db.preferences.get_for_user(user)

        identified = await self._identify_preference_topics(conversation, existing, run_id)
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
        classified = await self._classify_preference_valence(survivor_topics, conversation, run_id)
        if not classified:
            return bool(identified.existing)

        embedding_lookup = {topic.lower(): emb for topic, emb in survivors}
        self._store_classified_preferences(user, classified, embedding_lookup)
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

    async def _identify_preference_topics(
        self, conversation: str, existing: list, run_id: str
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
                agent_name=self.name,
                prompt_type=HistoryPromptType.PREFERENCE_IDENTIFICATION,
                run_id=run_id,
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
        lines = "\n".join(f"- {pref.valence}: {pref.content}" for pref in existing)
        return (
            f"Already known preferences (do NOT re-extract these or rephrasings of these):\n{lines}"
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
            assert matched.id is not None
            self.db.preferences.increment_mention_count(matched.id)
            already_bumped.add(matched.id)
            logger.info(
                "Preference '%s' mention count incremented (matches '%s')",
                topic[:50],
                matched.content[:50],
            )
        else:
            logger.debug("Skipping intra-batch duplicate: '%s'", topic[:50])

    # ── Pass 2: valence classification ───────────────────────────────────

    async def _classify_preference_valence(
        self, topics: list[str], conversation: str, run_id: str
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
                agent_name=self.name,
                prompt_type=HistoryPromptType.PREFERENCE_VALENCE,
                run_id=run_id,
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
                embedding=embedding,
                source=PennyConstants.PreferenceSource.EXTRACTED,
            )
            logger.info("Preference stored for %s: %s (%s)", user, content[:50], pref.valence)

    # ── Reaction handling ────────────────────────────────────────────────

    def _process_reactions(self, reactions: list) -> None:
        """Set thought valence for reactions to thought notifications.

        Reactions to regular messages are discarded — preference extraction
        runs only on text messages. All reactions are marked processed by the caller.
        """
        for reaction in reactions:
            parent = self.db.messages.get_by_id(reaction.parent_id) if reaction.parent_id else None
            if not parent or parent.thought_id is None:
                continue
            int_valence = self._emoji_to_int_valence(reaction.content)
            if int_valence is not None:
                self.db.thoughts.set_valence(parent.thought_id, int_valence)

    @staticmethod
    def _emoji_to_int_valence(emoji: str) -> int | None:
        """Classify an emoji as 1 (positive), -1 (negative), or None (unknown)."""
        if emoji in PennyConstants.POSITIVE_REACTION_EMOJIS:
            return 1
        if emoji in PennyConstants.NEGATIVE_REACTION_EMOJIS:
            return -1
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

    @staticmethod
    def _format_messages(messages: list) -> str:
        """Format user messages for the summarization prompt."""
        lines: list[str] = []
        for msg in messages:
            ts = msg.timestamp.strftime("%H:%M")
            lines.append(f"[{ts}] {msg.content}")
        return "\n".join(lines)
