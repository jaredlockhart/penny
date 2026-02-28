"""Notification agent — owns all proactive messaging to users.

Decoupled from extraction: the extraction pipeline stores facts silently,
and this agent scores entities by explicit user signals (emoji reactions,
follow-up questions, mentions) — filtering out noisy batch signals from
/learn sessions. Entities with negative emoji reactions are hard-vetoed,
and notification fatigue penalizes over-notified entities.

Gated by per-user exponential backoff and a per-entity cooldown.
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
from penny.constants import PennyConstants
from penny.database.models import Engagement, Entity, Event, Fact, FollowPrompt, LearnPrompt
from penny.interest import compute_interest_score
from penny.ollama.embeddings import deserialize_embedding, find_similar
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class NotificationAgent(Agent):
    """Background agent that sends proactive notifications to users.

    Three notification streams:
    1. Learn completion announcements — exclusive, bypass backoff
    2. Event notifications — per-follow-prompt cadence (independent loop)
    3. Fact discovery notifications — per-user exponential backoff (independent loop)

    Event and fact notifications run independently so events don't starve facts.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None
        self._backoff_state: dict[str, BackoffState] = {}
        self._last_notified_learn_prompt_id: dict[str, int | None] = {}
        self._last_notification: dict[str, tuple[int, datetime]] = {}  # user → (entity_id, sent_at)

    @property
    def name(self) -> str:
        return "notification"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending notifications."""
        self._channel = channel

    async def execute(self) -> bool:
        """Run notification loops for all users.

        Learn completions are exclusive (return immediately). Event and fact
        notifications run independently — both can fire in the same cycle.
        Returns True if any notification was sent.
        """
        if not self._channel:
            return False

        any_sent = False
        users = self.db.users.get_all_senders()
        for user in users:
            if self.db.users.is_muted(user):
                continue
            # Learn completion announcements take priority and bypass backoff
            if await self._try_learn_completion(user):
                return True
            # Event and fact notifications run independently
            if await self._try_event_notification(user):
                any_sent = True
            if await self._try_notify_user(user):
                any_sent = True

        return any_sent

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
        """Check for a due follow prompt and send one event notification.

        Each follow prompt has its own cadence (hourly/daily/weekly). Finds
        the most overdue prompt, picks the best unnotified event for it,
        sends the notification, and updates the prompt's last_notified_at.
        Independent of fact notification backoff.
        """
        prompt = self._find_due_follow_prompt(user)
        if prompt is None:
            return False

        assert prompt.id is not None
        unnotified = self.db.events.get_unnotified_for_follow_prompt(prompt.id)
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
        self.db.follow_prompts.update_last_notified(prompt.id)
        return True

    def _find_due_follow_prompt(self, user: str) -> FollowPrompt | None:
        """Find the most overdue follow prompt for a user.

        Iterates active follow prompts, checks if enough time has elapsed
        since last_notified_at based on each prompt's cadence. Returns the
        prompt with the greatest overdue ratio, or None if none are due.
        """
        follows = self.db.follow_prompts.get_active(user)
        if not follows:
            return None

        best: FollowPrompt | None = None
        best_overdue = 0.0

        default_seconds = PennyConstants.FOLLOW_CADENCE_SECONDS[
            PennyConstants.FOLLOW_DEFAULT_CADENCE
        ]

        for fp in follows:
            cadence_seconds = PennyConstants.FOLLOW_CADENCE_SECONDS.get(fp.cadence, default_seconds)

            if fp.last_notified_at is None:
                # Never notified — always due immediately
                overdue_ratio = float("inf")
            else:
                last = fp.last_notified_at
                if last.tzinfo is None:
                    last = last.replace(tzinfo=UTC)
                elapsed = (datetime.now(UTC) - last).total_seconds()
                if elapsed < cadence_seconds:
                    continue
                overdue_ratio = elapsed / cadence_seconds

            if overdue_ratio > best_overdue:
                best_overdue = overdue_ratio
                best = fp

        if best is not None:
            logger.debug(
                "Event notification: follow prompt '%s' is due (%.1fx overdue, cadence=%s)",
                best.prompt_text[:50],
                best_overdue,
                best.cadence,
            )

        return best

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
        result = await self._compose_user_facing(prompt, image_query=event.headline)

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

        Scores entities by explicit user signals, randomly picks from the
        top-N pool (respecting emoji veto, fatigue, cooldown, and learn-topic
        dedup), and announces all unnotified facts for the selected entity.
        Returns True if a notification was sent.
        """
        if not self._should_send(user):
            return False

        # Auto-tuning: record ignored notification before picking the next one
        self._record_ignored_notification(user)

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

        # Track this notification for auto-tuning outcome check
        self._last_notification[user] = (entity.id, datetime.now(UTC))

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
        """Score entities by explicit user signals + neighbor boost.

        Base: (filtered_interest + log2(fact_count + 1)) / fatigue
        Neighbor boost: base * (1 + factor * mean_neighbor_interest)
        """
        engagements_by_entity = self._build_engagement_map(user)
        all_interest = self._build_interest_map(engagements_by_entity)
        embedding_candidates = self._build_embedding_candidates(user)

        scored: list[tuple[float, Entity]] = []
        for eid, facts in facts_by_entity.items():
            entity = self.db.entities.get(eid)
            if entity is None:
                continue
            if not self._is_eligible(
                eid, entity, facts, engagements_by_entity, last_notified_learn_prompt_id
            ):
                continue

            base = self._compute_base_score(eid, facts, engagements_by_entity)
            boost = self._compute_neighbor_boost(eid, entity, all_interest, embedding_candidates)
            score = base * (1.0 + self.config.runtime.NOTIFICATION_NEIGHBOR_FACTOR * boost)
            scored.append((score, entity))

        return scored

    def _build_engagement_map(self, user: str) -> dict[int, list[Engagement]]:
        """Group all user engagements by entity ID."""
        result: dict[int, list[Engagement]] = defaultdict(list)
        for eng in self.db.engagements.get_for_user(user):
            if eng.entity_id is not None:
                result[eng.entity_id].append(eng)
        return result

    def _build_interest_map(
        self, engagements_by_entity: dict[int, list[Engagement]]
    ) -> dict[int, float]:
        """Precompute notification interest score for every entity with engagements."""
        result: dict[int, float] = {}
        for eid, engs in engagements_by_entity.items():
            notification_engs = [
                e for e in engs if e.engagement_type in PennyConstants.NOTIFICATION_ENGAGEMENT_TYPES
            ]
            result[eid] = compute_interest_score(
                notification_engs,
                half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
            )
        return result

    def _build_embedding_candidates(self, user: str) -> list[tuple[int, list[float]]]:
        """Build (entity_id, embedding_vector) pairs for neighbor search."""
        return [
            (e.id, deserialize_embedding(e.embedding))
            for e in self.db.entities.get_with_embeddings(user)
            if e.id is not None and e.embedding is not None
        ]

    def _is_eligible(
        self,
        eid: int,
        entity: Entity,
        facts: list[Fact],
        engagements_by_entity: dict[int, list[Engagement]],
        last_notified_learn_prompt_id: int | None,
    ) -> bool:
        """Check veto, cooldown, and learn-topic dedup filters."""
        if self._is_vetoed_by_emoji(eid, engagements_by_entity):
            return False
        if self._is_entity_on_cooldown(entity):
            return False
        if self._is_same_learn_topic(facts, last_notified_learn_prompt_id):
            logger.debug(
                "Notification: skipping '%s' (same learn topic as last notification, "
                "learn_prompt_id=%d)",
                entity.name,
                last_notified_learn_prompt_id,
            )
            return False
        return True

    def _compute_base_score(
        self,
        eid: int,
        facts: list[Fact],
        engagements_by_entity: dict[int, list[Engagement]],
    ) -> float:
        """Compute base score: (filtered_interest + log2(fact_count + 1)) / fatigue."""
        notification_engs = [
            eng
            for eng in engagements_by_entity.get(eid, [])
            if eng.engagement_type in PennyConstants.NOTIFICATION_ENGAGEMENT_TYPES
        ]
        interest = compute_interest_score(
            notification_engs,
            half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
        )
        fatigue = self._compute_notification_fatigue(eid)
        return (interest + math.log2(len(facts) + 1)) / fatigue

    def _compute_neighbor_boost(
        self,
        eid: int,
        entity: Entity,
        all_interest: dict[int, float],
        embedding_candidates: list[tuple[int, list[float]]],
    ) -> float:
        """Compute mean neighbor interest weighted by similarity.

        Finds the K most similar entities by embedding cosine similarity,
        filters to those with non-zero interest (engaged-only), and returns
        the mean of (interest * similarity). Bidirectional: positive neighbors
        lift, negative neighbors drag down.
        """
        if entity.embedding is None or not embedding_candidates:
            return 0.0

        query_vec = deserialize_embedding(entity.embedding)
        k = int(self.config.runtime.NOTIFICATION_NEIGHBOR_K)
        min_sim = self.config.runtime.NOTIFICATION_NEIGHBOR_MIN_SIMILARITY
        # Request k+1 to account for self-match (filtered out below)
        neighbors = find_similar(query_vec, embedding_candidates, top_k=k + 1, threshold=min_sim)

        # Filter to engaged-only (non-zero interest)
        engaged = [
            (nid, sim) for nid, sim in neighbors if nid != eid and all_interest.get(nid, 0.0) != 0.0
        ]
        if not engaged:
            return 0.0

        weighted = [all_interest[nid] * sim for nid, sim in engaged]
        return sum(weighted) / len(weighted)

    def _is_vetoed_by_emoji(
        self, entity_id: int, engagements_by_entity: dict[int, list[Engagement]]
    ) -> bool:
        """Check if an entity has any negative emoji reaction (hard veto)."""
        for eng in engagements_by_entity.get(entity_id, []):
            if (
                eng.engagement_type == PennyConstants.EngagementType.EMOJI_REACTION
                and eng.valence == PennyConstants.EngagementValence.NEGATIVE
            ):
                logger.debug("Notification: vetoed '%d' (negative emoji reaction)", entity_id)
                return True
        return False

    def _compute_notification_fatigue(self, entity_id: int) -> float:
        """Compute fatigue divisor based on how many facts have been notified."""
        total_notified = self.db.facts.count_notified(entity_id)
        return math.log2(total_notified + 2)

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

    def _record_ignored_notification(self, user: str) -> None:
        """Check if the previous notification was ignored and record a negative signal.

        If the last notification for this user received no engagement (no emoji,
        no follow-up, no mention), create a NOTIFICATION_IGNORED engagement to
        gradually suppress uninteresting entities.
        """
        prev = self._last_notification.get(user)
        if prev is None:
            return

        entity_id, sent_at = prev
        if self.db.engagements.has_engagement_since(entity_id, sent_at):
            return

        strength = self.config.runtime.NOTIFICATION_IGNORE_STRENGTH
        self.db.engagements.add(
            user=user,
            engagement_type=PennyConstants.EngagementType.NOTIFICATION_IGNORED,
            valence=PennyConstants.EngagementValence.NEGATIVE,
            strength=strength,
            entity_id=entity_id,
        )
        entity = self.db.entities.get(entity_id)
        entity_name = entity.name if entity else str(entity_id)
        logger.info(
            "Auto-tuning: recorded ignored notification for '%s' (strength=%.2f)",
            entity_name,
            strength,
        )

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
