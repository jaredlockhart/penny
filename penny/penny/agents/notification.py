"""Notification agent — owns all proactive messaging to users.

Decoupled from extraction: the extraction pipeline stores facts silently,
and this agent selects the most interesting un-notified discovery to
surface on each cycle, gated by per-user exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime

from penny.agents.base import Agent
from penny.channels.base import MessageChannel
from penny.database.models import Fact, LearnPrompt
from penny.interest import compute_interest_score
from penny.prompts import Prompt
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class _UserBackoff:
    """Per-user backoff state for proactive notifications."""

    __slots__ = ("last_proactive_send", "backoff_seconds")

    def __init__(self) -> None:
        self.last_proactive_send: datetime | None = None
        self.backoff_seconds: float = 0.0


class NotificationAgent(Agent):
    """Background agent that sends interest-ranked fact discovery notifications.

    Queries for un-notified facts, groups by entity, picks the highest-interest
    entity, composes a message, and sends it — one notification per cycle.

    Also sends learn completion announcements when a /learn topic finishes
    extraction, summarizing discovered entities with fact counts and interest scores.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None
        self._backoff_state: dict[str, _UserBackoff] = {}

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

        users = self.db.get_all_senders()
        for user in users:
            # Learn completion announcements take priority and bypass backoff
            if await self._try_learn_completion(user):
                return True
            if await self._try_notify_user(user):
                return True

        return False

    # --- Learn completion announcements ---

    async def _try_learn_completion(self, user: str) -> bool:
        """Check for completed learn prompts and send completion announcements.

        Bypasses backoff entirely — the user explicitly requested this research.
        Sends ALL ready announcements in one cycle (no one-per-cycle limit).
        Does NOT affect backoff state for entity notifications.
        Returns True if any announcement was sent.
        """
        assert self._channel is not None

        learn_prompts = self.db.get_unannounced_completed_learn_prompts(user)
        if not learn_prompts:
            return False

        any_sent = False
        for lp in learn_prompts:
            assert lp.id is not None
            search_logs = self.db.get_search_logs_by_learn_prompt(lp.id)

            if not search_logs:
                # No searches were made (e.g., no search tool) — mark announced, skip
                self.db.mark_learn_prompt_announced(lp.id)
                continue

            if not all(sl.extracted for sl in search_logs):
                continue  # Extraction not finished yet

            sent = await self._send_learn_completion(lp, user)
            if not sent:
                continue

            # Mark remaining un-notified facts from this learn prompt as notified
            search_log_ids = [sl.id for sl in search_logs if sl.id is not None]
            facts = self.db.get_facts_by_search_log_ids(search_log_ids)
            unnotified_ids = [f.id for f in facts if f.notified_at is None and f.id is not None]
            if unnotified_ids:
                self.db.mark_facts_notified(unnotified_ids)

            self.db.mark_learn_prompt_announced(lp.id)
            any_sent = True

        return any_sent

    async def _send_learn_completion(self, lp: LearnPrompt, user: str) -> bool:
        """Compose and send a learn completion summary via the model.

        Returns True if the announcement was sent.
        """
        assert self._channel is not None
        assert lp.id is not None

        search_logs = self.db.get_search_logs_by_learn_prompt(lp.id)
        search_log_ids = [sl.id for sl in search_logs if sl.id is not None]
        facts = self.db.get_facts_by_search_log_ids(search_log_ids)

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

        # Group facts by entity, sorted by interest score
        facts_by_entity: dict[int, list[Fact]] = defaultdict(list)
        for fact in facts:
            facts_by_entity[fact.entity_id].append(fact)

        all_engagements = self.db.get_user_engagements(user)
        engagements_by_entity: dict[int, list] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        # Build entity sections with actual facts for the model
        entity_sections: list[tuple[float, str]] = []
        for entity_id, entity_facts in facts_by_entity.items():
            entity = self.db.get_entity(entity_id)
            if entity is None:
                continue
            score = compute_interest_score(
                engagements_by_entity.get(entity_id, []),
                half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
            )
            facts_text = "\n".join(f"- {f.content}" for f in entity_facts)
            section = f"{entity.name}:\n{facts_text}"
            entity_sections.append((score, section))

        entity_sections.sort(key=lambda x: x[0], reverse=True)
        all_sections = "\n\n".join(section for _, section in entity_sections)

        prompt = (
            f"{Prompt.LEARN_COMPLETION_SUMMARY_PROMPT.format(topic=lp.prompt_text)}"
            f"\n\nEntities and facts discovered:\n\n{all_sections}"
        )

        result = await self._compose_user_facing(prompt)
        if not result.answer:
            logger.warning("Failed to compose learn completion for '%s'", lp.prompt_text)
            return False

        typing_task = asyncio.create_task(self._channel._typing_loop(user))
        try:
            await self._channel.send_response(
                user,
                result.answer,
                parent_id=None,
                attachments=result.attachments or None,
            )
            logger.info(
                "Learn completion announcement sent for '%s' to %s",
                lp.prompt_text,
                user,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(user, False)

        return True

    # --- Fact discovery notifications ---

    async def _try_notify_user(self, user: str) -> bool:
        """Attempt to send one notification to this user.

        Returns True if a notification was sent.
        """
        if not self._should_send(user):
            return False

        unnotified = self.db.get_unnotified_facts(user)
        if not unnotified:
            return False

        # Group facts by entity
        facts_by_entity: dict[int, list] = defaultdict(list)
        for fact in unnotified:
            facts_by_entity[fact.entity_id].append(fact)

        # Pick highest-interest entity
        entity_id = self._pick_best_entity(user, list(facts_by_entity.keys()))
        if entity_id is None:
            return False

        entity = self.db.get_entity(entity_id)
        if entity is None:
            return False

        facts = facts_by_entity[entity_id]

        # Determine if this is a new entity (no previously-notified facts)
        all_facts = self.db.get_entity_facts(entity_id)
        is_new = all(f.notified_at is None for f in all_facts)

        # Compose and send
        sent = await self._send_notification(user, entity, facts, is_new)
        if not sent:
            return False

        # Mark these facts as notified
        fact_ids = [f.id for f in facts if f.id is not None]
        self.db.mark_facts_notified(fact_ids)

        # Update backoff
        self._mark_proactive_sent(user)

        return True

    def _should_send(self, user: str) -> bool:
        """Check if we should send proactive notifications to this user."""
        state = self._backoff_state.get(user)
        if state is None:
            return True

        # Check if user has interacted (message or command) since our last proactive send
        if state.last_proactive_send is not None:
            latest_interaction = self.db.get_latest_user_interaction_time(user)
            if latest_interaction is not None:
                interaction_time = latest_interaction
                if interaction_time.tzinfo is None:
                    interaction_time = interaction_time.replace(tzinfo=UTC)
                last_send = state.last_proactive_send
                if last_send.tzinfo is None:
                    last_send = last_send.replace(tzinfo=UTC)
                if interaction_time > last_send:
                    state.backoff_seconds = 0.0

        if state.backoff_seconds <= 0:
            return True

        if state.last_proactive_send is None:
            return True

        now = datetime.now(UTC)
        last_send = state.last_proactive_send
        if last_send.tzinfo is None:
            last_send = last_send.replace(tzinfo=UTC)
        elapsed = (now - last_send).total_seconds()
        if elapsed >= state.backoff_seconds:
            # Backoff expired — return to eager state so next cycle starts fresh
            state.backoff_seconds = 0.0
            return True
        return False

    def _mark_proactive_sent(self, user: str) -> None:
        """Record that we sent a notification and increase backoff."""
        state = self._backoff_state.get(user)
        if state is None:
            state = _UserBackoff()
            self._backoff_state[user] = state

        state.last_proactive_send = datetime.now(UTC)
        if state.backoff_seconds <= 0:
            state.backoff_seconds = self.config.runtime.NOTIFICATION_INITIAL_BACKOFF
        else:
            state.backoff_seconds = min(
                state.backoff_seconds * 2,
                self.config.runtime.NOTIFICATION_MAX_BACKOFF,
            )

    def _pick_best_entity(self, user: str, entity_ids: list[int]) -> int | None:
        """Pick the entity with the highest interest score from candidates."""
        if not entity_ids:
            return None

        all_engagements = self.db.get_user_engagements(user)
        engagements_by_entity: dict[int, list] = defaultdict(list)
        for eng in all_engagements:
            if eng.entity_id is not None:
                engagements_by_entity[eng.entity_id].append(eng)

        best_id: int | None = None
        best_score = float("-inf")
        for eid in entity_ids:
            score = compute_interest_score(
                engagements_by_entity.get(eid, []),
                half_life_days=self.config.runtime.INTEREST_SCORE_HALF_LIFE_DAYS,
            )
            if score > best_score:
                best_score = score
                best_id = eid

        return best_id

    def _get_learn_topic(self, facts: list[Fact]) -> str | None:
        """Trace facts back to a /learn prompt topic, if any.

        Follows: Fact.source_search_log_id → SearchLog.learn_prompt_id → LearnPrompt.prompt_text
        Returns the prompt_text of the first matching learn prompt, or None.
        """
        for fact in facts:
            if fact.source_search_log_id is None:
                continue
            search_log = self.db.get_search_log(fact.source_search_log_id)
            if search_log is None or search_log.learn_prompt_id is None:
                continue
            learn_prompt = self.db.get_learn_prompt(search_log.learn_prompt_id)
            if learn_prompt is not None:
                return learn_prompt.prompt_text
        return None

    async def _send_notification(
        self, user: str, entity: object, facts: list[Fact], is_new: bool
    ) -> bool:
        """Compose and send a notification for one entity's new facts."""
        assert self._channel is not None

        facts_text = "\n".join(f"- {fact.content}" for fact in facts)
        learn_topic = self._get_learn_topic(facts)

        if learn_topic:
            if is_new:
                prompt_template = Prompt.FACT_DISCOVERY_NEW_ENTITY_LEARN_PROMPT
            else:
                prompt_template = Prompt.FACT_DISCOVERY_KNOWN_ENTITY_LEARN_PROMPT
            prompt_text = prompt_template.format(
                entity_name=entity.name,  # type: ignore[union-attr]
                learn_topic=learn_topic,
            )
        else:
            if is_new:
                prompt_template = Prompt.FACT_DISCOVERY_NEW_ENTITY_PROMPT
            else:
                prompt_template = Prompt.FACT_DISCOVERY_KNOWN_ENTITY_PROMPT
            prompt_text = prompt_template.format(entity_name=entity.name)  # type: ignore[union-attr]

        prompt = f"{prompt_text}\n\nNew facts:\n{facts_text}"

        result = await self._compose_user_facing(
            prompt,
            image_query=entity.name,  # type: ignore[union-attr]
        )
        if not result.answer:
            return False
        if len(result.answer) < self.config.runtime.NOTIFICATION_MIN_LENGTH:
            logger.debug(
                "Skipping near-empty notification (%d chars): %r",
                len(result.answer),
                result.answer,
            )
            return False

        typing_task = asyncio.create_task(self._channel._typing_loop(user))
        try:
            await self._channel.send_response(
                user,
                result.answer,
                parent_id=None,
                attachments=result.attachments or None,
            )
            logger.info(
                "Notification sent for entity '%s' (%d facts) to %s",
                entity.name,  # type: ignore[union-attr]
                len(facts),
                user,
            )
        finally:
            typing_task.cancel()
            await self._channel.send_typing(user, False)

        return True
