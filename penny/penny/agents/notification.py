"""Notification agent — owns all proactive messaging to users.

Decoupled from extraction: the extraction pipeline stores facts silently,
and this agent scores entities by interest + enrichment volume, then randomly
selects from the top-N pool — gated by per-user exponential backoff and a
per-entity cooldown.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import defaultdict
from datetime import UTC, datetime

from penny.agents.backoff import BackoffState
from penny.agents.base import Agent
from penny.channels.base import MessageChannel
from penny.database.models import Engagement, Entity, Event, Fact, LearnPrompt
from penny.interest import compute_interest_score
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class NotificationAgent(Agent):
    """Background agent that sends proactive notifications to users.

    Three notification streams, checked in priority order:
    1. Learn completion announcements (bypass backoff)
    2. Event notifications — news about followed topics (respect backoff)
    3. Fact discovery notifications — enrichment findings (respect backoff)

    One notification per cycle maximum.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None
        self._backoff_state: dict[str, BackoffState] = {}
        self._last_notified_learn_prompt_id: dict[str, int | None] = {}

    @property
    def name(self) -> str:
        return "notification"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending notifications."""
        self._channel = channel

    async def execute(self) -> bool:
        """Send at most one notification across all users.

        Returns True if a notification was sent.
        """
        if not self._channel:
            return False

        users = self.db.users.get_all_senders()
        for user in users:
            if self.db.users.is_muted(user):
                continue
            # Learn completion announcements take priority and bypass backoff
            if await self._try_learn_completion(user):
                return True
            # Event notifications (news about followed topics)
            if await self._try_event_notification(user):
                return True
            if await self._try_notify_user(user):
                return True

        return False

    # --- Learn completion announcements ---

    async def _try_learn_completion(self, user: str) -> bool:
        """Check for completed learn prompts and send one completion announcement.

        Bypasses backoff entirely — the user explicitly requested this research.
        Sends ONE announcement per cycle so multiple completions arrive spaced out
        rather than in a burst. Does NOT affect backoff state for entity notifications.
        Returns True if an announcement was sent.
        """
        assert self._channel is not None

        learn_prompts = self.db.learn_prompts.get_unannounced_completed(user)
        if not learn_prompts:
            return False

        for lp in learn_prompts:
            assert lp.id is not None
            search_logs = self.db.searches.get_by_learn_prompt(lp.id)

            if not search_logs:
                # No searches were made (e.g., no search tool) — mark announced, skip
                self.db.learn_prompts.mark_announced(lp.id)
                continue

            if not all(sl.extracted for sl in search_logs):
                continue  # Extraction not finished yet

            sent = await self._send_learn_completion(lp, user)
            if not sent:
                continue

            # Mark remaining un-notified facts from this learn prompt as notified
            search_log_ids = [sl.id for sl in search_logs if sl.id is not None]
            facts = self.db.facts.get_by_search_log_ids(search_log_ids)
            unnotified_ids = [f.id for f in facts if f.notified_at is None and f.id is not None]
            if unnotified_ids:
                self.db.facts.mark_notified(unnotified_ids)

            self.db.learn_prompts.mark_announced(lp.id)
            return True

        return False

    async def _send_learn_completion(self, lp: LearnPrompt, user: str) -> bool:
        """Compose and send a learn completion summary via the model.

        Returns True if the announcement was sent.
        """
        assert self._channel is not None
        assert lp.id is not None

        facts = self._get_learn_prompt_facts(lp.id)

        if not facts:
            message = (
                f"{PennyResponse.LEARN_COMPLETE_HEADER.format(topic=lp.prompt_text)}"
                f"\n\n{PennyResponse.LEARN_COMPLETE_NO_ENTITIES}"
            )
            await self._channel.send_response(user, message, parent_id=None)
            logger.info(
                "Learn completion announcement sent (no entities) for '%s' to %s",
                lp.prompt_text,
                user,
            )
            return True

        prompt = self._build_learn_completion_prompt(lp, facts, user)
        result = await self._compose_user_facing(prompt, image_query=lp.prompt_text)
        if not result.answer:
            logger.warning("Failed to compose learn completion for '%s'", lp.prompt_text)
            return False

        await self._send_with_typing(user, result.answer, result.attachments)
        logger.info(
            "Learn completion announcement sent for '%s' to %s",
            lp.prompt_text,
            user,
        )
        return True

    def _get_learn_prompt_facts(self, learn_prompt_id: int) -> list[Fact]:
        """Fetch all facts associated with a learn prompt's search logs."""
        search_logs = self.db.searches.get_by_learn_prompt(learn_prompt_id)
        search_log_ids = [sl.id for sl in search_logs if sl.id is not None]
        return self.db.facts.get_by_search_log_ids(search_log_ids)

    def _build_learn_completion_prompt(self, lp: LearnPrompt, facts: list[Fact], user: str) -> str:
        """Build the LLM prompt for a learn completion announcement."""
        entity_sections = self._group_facts_by_scored_entity(facts, user)
        entity_sections.sort(key=lambda x: x[0], reverse=True)
        all_sections = "\n\n".join(section for _, section in entity_sections)

        return (
            f"{Prompt.LEARN_COMPLETION_SUMMARY_PROMPT.format(topic=lp.prompt_text)}"
            f"\n\nEntities and facts discovered:\n\n{all_sections}"
        )

    def _group_facts_by_scored_entity(
        self, facts: list[Fact], user: str
    ) -> list[tuple[float, str]]:
        """Group facts by entity and score each group by interest."""
        facts_by_entity: dict[int, list[Fact]] = defaultdict(list)
        for fact in facts:
            facts_by_entity[fact.entity_id].append(fact)

        all_engagements = self.db.engagements.get_for_user(user)
        engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        sections: list[tuple[float, str]] = []
        for entity_id, entity_facts in facts_by_entity.items():
            entity = self.db.entities.get(entity_id)
            if entity is None:
                continue
            score = compute_interest_score(
                engagements_by_entity.get(entity_id, []),
                half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
            )
            facts_text = "\n".join(f"- {f.content}" for f in entity_facts)
            sections.append((score, f"{entity.name}:\n{facts_text}"))

        return sections

    # --- Event notifications ---

    async def _try_event_notification(self, user: str) -> bool:
        """Check for unnotified events and send one notification.

        Scores events by linked entity interest + timeliness decay, picks
        the highest-scoring event, composes a message, and sends it.
        Respects per-user backoff. Returns True if sent.
        """
        if not self._should_send(user):
            return False

        unnotified = self.db.events.get_unnotified(user)
        if not unnotified:
            return False

        event = self._pick_best_event(user, unnotified)
        if event is None:
            return False

        sent = await self._send_event_notification(user, event)
        if not sent:
            return False

        assert event.id is not None
        self.db.events.mark_notified([event.id])
        self._mark_proactive_sent(user)
        return True

    def _pick_best_event(self, user: str, events: list[Event]) -> Event | None:
        """Score events by entity interest + timeliness, return the best one."""
        all_engagements = self.db.engagements.get_for_user(user)
        engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        scored: list[tuple[float, Event]] = []
        for event in events:
            score = self._score_event(event, engagements_by_entity)
            scored.append((score, event))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_event = scored[0]

        logger.info(
            "Event notification: picked '%s' (score=%.2f) from %d unnotified",
            best_event.headline[:50],
            best_score,
            len(scored),
        )
        return best_event

    def _score_event(
        self,
        event: Event,
        engagements_by_entity: dict[int, list[Engagement]],
    ) -> float:
        """Score an event: sum of linked entity interest + timeliness bonus."""
        assert event.id is not None

        # Entity interest component
        linked_entities = self.db.events.get_entities_for_event(event.id)
        interest_total = 0.0
        for entity in linked_entities:
            assert entity.id is not None
            interest_total += compute_interest_score(
                engagements_by_entity.get(entity.id, []),
                half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
            )

        # Timeliness component: 2^(-hours_since / half_life)
        timeliness = self._compute_timeliness(event)

        return interest_total + timeliness

    def _compute_timeliness(self, event: Event) -> float:
        """Compute timeliness decay: 2^(-hours_since / half_life)."""
        half_life = self.config.runtime.EVENT_TIMELINESS_HALF_LIFE_HOURS
        occurred = event.occurred_at
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=UTC)
        hours_since = (datetime.now(UTC) - occurred).total_seconds() / 3600.0
        if hours_since < 0:
            hours_since = 0.0
        return math.pow(2.0, -hours_since / half_life)

    async def _send_event_notification(self, user: str, event: Event) -> bool:
        """Compose and send a notification about an event."""
        assert self._channel is not None

        prompt = self._build_event_prompt(event)
        result = await self._compose_user_facing(prompt)

        if not result.answer:
            return False
        if len(result.answer) < self.config.runtime.NOTIFICATION_MIN_LENGTH:
            logger.debug(
                "Skipping short event notification (%d chars): %r",
                len(result.answer),
                result.answer,
            )
            return False

        await self._send_with_typing(user, result.answer, result.attachments)
        logger.info(
            "Event notification sent for '%s' to %s",
            event.headline[:50],
            user,
        )
        return True

    def _build_event_prompt(self, event: Event) -> str:
        """Build the LLM prompt for an event notification."""
        parts = [Prompt.EVENT_NOTIFICATION_PROMPT, ""]
        parts.append(f"Headline: {event.headline}")
        if event.summary:
            parts.append(f"Summary: {event.summary}")
        if event.source_url:
            parts.append(f"Source: {event.source_url}")
        return "\n".join(parts)

    # --- Fact discovery notifications ---

    async def _try_notify_user(self, user: str) -> bool:
        """Attempt to send one notification to this user.

        Scores entities by interest + enrichment volume, randomly picks from
        the top-N pool (respecting per-entity cooldown and learn-topic dedup),
        and announces all unnotified facts for the selected entity.
        Returns True if a notification was sent.
        """
        if not self._should_send(user):
            return False

        unnotified = self.db.facts.get_unnotified(user)
        if not unnotified:
            return False

        # Group facts by entity
        facts_by_entity: dict[int, list[Fact]] = defaultdict(list)
        for fact in unnotified:
            facts_by_entity[fact.entity_id].append(fact)

        # Score and randomly select from top-N pool
        last_learn_prompt_id = self._last_notified_learn_prompt_id.get(user)
        entity = self._pick_scored_entity(user, facts_by_entity, last_learn_prompt_id)
        if entity is None:
            return False
        assert entity.id is not None

        facts = facts_by_entity[entity.id]

        # Determine if this is a new entity (no previously-notified facts)
        all_facts = self.db.facts.get_for_entity(entity.id)
        is_new = all(f.notified_at is None for f in all_facts)

        # Compose and send
        sent = await self._send_notification(user, entity, facts, is_new)
        if not sent:
            return False

        # Mark these facts as notified + record entity notification time
        fact_ids = [f.id for f in facts if f.id is not None]
        self.db.facts.mark_notified(fact_ids)
        self.db.entities.update_last_notified_at(entity.id)

        # Track which learn prompt this notification came from (to suppress same-topic dedup)
        self._last_notified_learn_prompt_id[user] = self._get_learn_prompt_id(facts)

        # Update backoff
        self._mark_proactive_sent(user)

        return True

    def _pick_scored_entity(
        self,
        user: str,
        facts_by_entity: dict[int, list[Fact]],
        last_notified_learn_prompt_id: int | None = None,
    ) -> Entity | None:
        """Score entities by interest + enrichment volume, pick randomly from top-N.

        Score formula: interest + log2(unannounced_count + 1)
        - Interest pulls high-engagement entities toward the top
        - log2(unannounced_count) rewards entities with more new material
        - Random selection from the top pool prevents stagnation
        - Per-entity cooldown ensures no entity dominates

        Filters: per-entity cooldown, learn-topic dedup (same as before).
        """
        pool_size = self.config.runtime.NOTIFICATION_POOL_SIZE
        scored = self._score_eligible_entities(user, facts_by_entity, last_notified_learn_prompt_id)

        if not scored:
            return None

        # Sort descending by score, take top-N, pick randomly
        scored.sort(key=lambda x: x[0], reverse=True)
        pool = scored[:pool_size]
        _, chosen = random.choice(pool)

        logger.info(
            "Notification: picked '%s' (score=%.2f) from pool of %d (top-%d of %d eligible)",
            chosen.name,
            next(s for s, e in pool if e is chosen),
            len(pool),
            pool_size,
            len(scored),
        )

        return chosen

    def _score_eligible_entities(
        self,
        user: str,
        facts_by_entity: dict[int, list[Fact]],
        last_notified_learn_prompt_id: int | None,
    ) -> list[tuple[float, Entity]]:
        """Score each entity by interest + enrichment volume, filtering ineligible ones."""
        all_engagements = self.db.engagements.get_for_user(user)
        engagements_by_entity: dict[int, list[Engagement]] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        scored: list[tuple[float, Entity]] = []
        for eid, facts in facts_by_entity.items():
            entity = self.db.entities.get(eid)
            if entity is None:
                continue
            if self._is_entity_on_cooldown(entity):
                continue
            if self._is_same_learn_topic(facts, last_notified_learn_prompt_id):
                logger.debug(
                    "Notification: skipping '%s' (same learn topic as last notification, "
                    "learn_prompt_id=%d)",
                    entity.name,
                    last_notified_learn_prompt_id,
                )
                continue

            interest = compute_interest_score(
                engagements_by_entity.get(eid, []),
                half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
            )
            score = interest + math.log2(len(facts) + 1)
            scored.append((score, entity))

        return scored

    def _is_entity_on_cooldown(self, entity: Entity) -> bool:
        """Check whether an entity was notified too recently."""
        if entity.last_notified_at is None:
            return False

        cooldown = self.config.runtime.NOTIFICATION_ENTITY_COOLDOWN
        now = datetime.now(UTC)
        last = entity.last_notified_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        elapsed = (now - last).total_seconds()

        if elapsed < cooldown:
            logger.debug(
                "Notification: skipping '%s' (notified %.0fs ago, cooldown=%.0fs)",
                entity.name,
                elapsed,
                cooldown,
            )
            return True
        return False

    def _is_same_learn_topic(
        self, facts: list[Fact], last_notified_learn_prompt_id: int | None
    ) -> bool:
        """Check if these facts share the same learn topic as the last notification."""
        if last_notified_learn_prompt_id is None:
            return False
        entity_learn_prompt_id = self._get_learn_prompt_id(facts)
        return entity_learn_prompt_id == last_notified_learn_prompt_id

    def _should_send(self, user: str) -> bool:
        """Check if we should send proactive notifications to this user."""
        state = self._backoff_state.get(user)
        if state is None:
            return True
        latest = self.db.messages.get_latest_interaction_time(user)
        return state.should_act(latest, self.config.runtime.NOTIFICATION_INITIAL_BACKOFF)

    def _mark_proactive_sent(self, user: str) -> None:
        """Record that we sent a notification and increase backoff."""
        state = self._backoff_state.get(user)
        if state is None:
            state = BackoffState()
            self._backoff_state[user] = state
        state.mark_done(
            self.config.runtime.NOTIFICATION_INITIAL_BACKOFF,
            self.config.runtime.NOTIFICATION_MAX_BACKOFF,
        )

    def _get_learn_prompt_id(self, facts: list[Fact]) -> int | None:
        """Trace facts back to a /learn prompt ID, if any.

        Follows: Fact.source_search_log_id → SearchLog.learn_prompt_id
        Returns the learn_prompt_id of the first matching search log, or None.
        """
        for fact in facts:
            if fact.source_search_log_id is None:
                continue
            search_log = self.db.searches.get(fact.source_search_log_id)
            if search_log is None or search_log.learn_prompt_id is None:
                continue
            return search_log.learn_prompt_id
        return None

    def _get_learn_topic(self, facts: list[Fact]) -> str | None:
        """Trace facts back to a /learn prompt topic, if any.

        Follows: Fact.source_search_log_id → SearchLog.learn_prompt_id → LearnPrompt.prompt_text
        Returns the prompt_text of the first matching learn prompt, or None.
        """
        for fact in facts:
            if fact.source_search_log_id is None:
                continue
            search_log = self.db.searches.get(fact.source_search_log_id)
            if search_log is None or search_log.learn_prompt_id is None:
                continue
            learn_prompt = self.db.learn_prompts.get(search_log.learn_prompt_id)
            if learn_prompt is not None:
                return learn_prompt.prompt_text
        return None

    async def _send_notification(
        self, user: str, entity: Entity, facts: list[Fact], is_new: bool
    ) -> bool:
        """Compose and send a notification for one entity's new facts."""
        assert self._channel is not None

        prompt = self._build_notification_prompt(entity, facts, is_new)
        image_query = f"{entity.name} {entity.tagline}" if entity.tagline else entity.name
        result = await self._compose_user_facing(prompt, image_query=image_query)

        if not result.answer:
            return False
        if len(result.answer) < self.config.runtime.NOTIFICATION_MIN_LENGTH:
            logger.debug(
                "Skipping near-empty notification (%d chars): %r",
                len(result.answer),
                result.answer,
            )
            return False

        await self._send_with_typing(user, result.answer, result.attachments)
        logger.info(
            "Notification sent for entity '%s' (%d facts) to %s",
            entity.name,
            len(facts),
            user,
        )
        return True

    def _build_notification_prompt(self, entity: Entity, facts: list[Fact], is_new: bool) -> str:
        """Build the LLM prompt for a fact discovery notification."""
        facts_text = "\n".join(f"- {fact.content}" for fact in facts)
        learn_topic = self._get_learn_topic(facts)

        if learn_topic:
            template = (
                Prompt.FACT_DISCOVERY_NEW_ENTITY_LEARN_PROMPT
                if is_new
                else Prompt.FACT_DISCOVERY_KNOWN_ENTITY_LEARN_PROMPT
            )
            prompt_text = template.format(entity_name=entity.name, learn_topic=learn_topic)
        else:
            template = (
                Prompt.FACT_DISCOVERY_NEW_ENTITY_PROMPT
                if is_new
                else Prompt.FACT_DISCOVERY_KNOWN_ENTITY_PROMPT
            )
            prompt_text = template.format(entity_name=entity.name)

        if entity.tagline:
            return (
                f"{prompt_text}\nContext: {entity.name} is {entity.tagline}."
                f"\n\nNew facts:\n{facts_text}"
            )
        return f"{prompt_text}\n\nNew facts:\n{facts_text}"

    async def _send_with_typing(
        self, user: str, text: str, attachments: list[str] | None = None
    ) -> None:
        """Send a message with a typing indicator active during delivery."""
        assert self._channel is not None

        typing_task = asyncio.create_task(self._channel._typing_loop(user))
        try:
            await self._channel.send_response(
                user,
                text,
                parent_id=None,
                attachments=attachments or None,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(user, False)
