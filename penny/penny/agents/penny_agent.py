"""PennyAgent — shared base for Penny's conversation and thinking agents.

Provides common tool building, context injection, interest profiles, and
entity/event retrieval that both ChatAgent and ThinkingAgent use.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.agents.models import MessageRole, ToolCallRecord
from penny.commands.models import CommandContext
from penny.constants import PennyConstants
from penny.database.models import Entity, Event
from penny.ollama.embeddings import deserialize_embedding, find_similar
from penny.ollama.similarity import embed_text
from penny.tools.base import Tool, ToolExecutor, ToolRegistry
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.follow import FollowTool
from penny.tools.learn import LearnTool
from penny.tools.message_user import MessageUserTool
from penny.tools.recall import RecallTool
from penny.tools.search import SearchTool

if TYPE_CHECKING:
    from penny.tools.news import NewsTool

logger = logging.getLogger(__name__)


class PennyAgent(Agent):
    """Shared base for Penny's mind — conversation and inner monologue."""

    THOUGHT_CONTEXT_LIMIT = 10

    def __init__(
        self,
        search_tool: Tool | None = None,
        news_tool: NewsTool | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._search_tool = search_tool
        self._news_tool = news_tool
        self._current_user: str | None = None

    # ── Tool building ──────────────────────────────────────────────────────

    def _build_tools(self, user: str) -> list[Tool]:
        """Build the shared tool list. Subclasses can extend via override."""
        searches = int(self.config.runtime.LEARN_PROMPT_DEFAULT_SEARCHES)
        tools: list[Tool] = [
            RecallTool(db=self.db, user=user),
            LearnTool(db=self.db, user=user, searches_remaining=searches),
        ]
        if self._search_tool:
            tools.append(self._search_tool)
        if self._news_tool:
            tools.append(FetchNewsTool(news_tool=self._news_tool))
            tools.append(FollowTool(command_context=self._build_command_context(user)))
        return tools

    def _build_command_context(self, user: str) -> CommandContext:
        """Build a CommandContext for delegating to command handlers."""
        return CommandContext(
            db=self.db,
            config=self.config,
            foreground_model_client=self._foreground_model_client,
            user=user,
            channel_type="",
            start_time=datetime.now(),
        )

    def _install_tools(self, tools: list[Tool]) -> None:
        """Replace the agent's tool registry and executor."""
        self._tool_registry = ToolRegistry()
        for tool in tools:
            self._tool_registry.register(tool)
        self._tool_executor = ToolExecutor(self._tool_registry, timeout=self.config.tool_timeout)

    # ── Context building ───────────────────────────────────────────────────

    async def _build_context(
        self,
        user: str,
        content: str | None = None,
    ) -> list[tuple[str, str]] | None:
        """Build shared context: identity → knowledge → conversation.

        Order: profile, interests, entity knowledge (anchored to user message),
        recent events, history summaries, timeline (messages + thoughts).
        """
        history = self._inject_profile_context(user, None, content)
        history = self._inject_interest_context(user, history)
        if content:
            history = await self._inject_entity_context(content, user, history)
        history = self._inject_recent_events(user, history)
        history = self._inject_history_context(user, history)
        history = self._inject_timeline_context(user, history)
        return history

    def _inject_profile_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
        content: str | None = None,
    ) -> list[tuple[str, str]] | None:
        """Prepend user profile context to history and configure search redaction."""
        try:
            user_info = self.db.users.get_info(sender)
            if not user_info:
                return history

            history = history or []
            history = [(MessageRole.USER.value, f"My name is {user_info.name}."), *history]
            logger.debug("Injected profile context for %s", sender)

            if content is not None:
                name = user_info.name
                user_said_name = bool(re.search(rf"\b{re.escape(name)}\b", content, re.IGNORECASE))
                if self._search_tool and isinstance(self._search_tool, SearchTool):
                    self._search_tool.redact_terms = [] if user_said_name else [name]
        except Exception:
            pass

        return history

    def _inject_timeline_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject today's messages and thoughts as individual entries, interleaved by time."""
        try:
            limit = int(self.config.runtime.MESSAGE_CONTEXT_LIMIT)
            midnight = self._midnight_today()
            messages = self.db.messages.get_messages_since(sender, since=midnight, limit=limit)
            all_thoughts = self.db.thoughts.get_recent(sender, limit=self.THOUGHT_CONTEXT_LIMIT)
            thoughts = [t for t in all_thoughts if t.created_at >= midnight]
            if not messages and not thoughts:
                return history

            entries: list[tuple[datetime, str, str]] = []
            for msg in messages:
                if msg.direction == PennyConstants.MessageDirection.INCOMING:
                    entries.append((msg.timestamp, MessageRole.USER.value, msg.content))
                else:
                    entries.append((msg.timestamp, MessageRole.SYSTEM.value, msg.content))
            for t in thoughts:
                entries.append((t.created_at, MessageRole.SYSTEM.value, f"I thought: {t.content}"))
            entries.sort(key=lambda e: e[0])

            history = history or []
            for _, role, content in entries:
                history.append((role, content))
            logger.debug(
                "Injected timeline context (%d messages, %d thoughts)", len(messages), len(thoughts)
            )
        except Exception:
            logger.warning("Timeline context retrieval failed, proceeding without")
        return history

    def _inject_history_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject daily conversation history summaries as individual entries."""
        try:
            limit = int(self.config.runtime.HISTORY_CONTEXT_LIMIT)
            entries = self.db.history.get_recent(
                sender, PennyConstants.HistoryDuration.DAILY, limit=limit
            )
            if not entries:
                return history

            history = history or []
            count = 0
            for entry in entries:
                for line in entry.topics.strip().splitlines():
                    line = line.strip().lstrip("- ").strip()
                    if line:
                        content = f"I remember we talked about {line}"
                        history.append((MessageRole.SYSTEM.value, content))
                        count += 1
            logger.debug("Injected history context (%d topics)", count)
        except Exception:
            logger.warning("History context retrieval failed, proceeding without")
        return history

    @staticmethod
    def _midnight_today() -> datetime:
        """Return midnight UTC for today as a naive datetime.

        Naive because SQLite strips timezone info — all stored datetimes are naive UTC.
        """
        return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    def _inject_interest_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject learn and follow topics as individual user-voice messages."""
        try:
            history = history or []
            count = 0
            learn_prompts = self.db.learn_prompts.get_for_user(sender)
            for lp in learn_prompts:
                history.append((MessageRole.USER.value, f"I'm interested in {lp.prompt_text}."))
                count += 1
            follow_prompts = self.db.follow_prompts.get_active(sender)
            for fp in follow_prompts:
                history.append((MessageRole.USER.value, f"Keep me updated on {fp.prompt_text}."))
                count += 1
            if count:
                logger.debug("Injected interest context (%d topics)", count)
        except Exception:
            logger.warning("Interest context retrieval failed, proceeding without")
        return history

    async def _inject_entity_context(
        self,
        content: str,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Retrieve entity knowledge as individual system messages per entity."""
        try:
            entity_messages = await self._retrieve_entity_messages(content, sender)
            if entity_messages:
                history = history or []
                history = [*history, *entity_messages]
                logger.debug("Injected entity context (%d entities)", len(entity_messages))
        except Exception:
            logger.warning("Entity context retrieval failed, proceeding without")
        return history

    def _inject_recent_events(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject recent events as individual system messages."""
        try:
            events = self.db.events.get_for_user(sender)
            if not events:
                return history
            recent = self._pick_latest_per_follow(events)
            follow_topics = self._load_follow_topics(sender)
            history = history or []
            for event in recent:
                content = self._format_event(event, follow_topics)
                history.append((MessageRole.SYSTEM.value, content))
            logger.debug("Injected event context (%d events)", len(recent))
        except Exception:
            logger.warning("Event context retrieval failed, proceeding without")
        return history

    @staticmethod
    def _pick_latest_per_follow(events: list[Event]) -> list[Event]:
        """Pick the most recent event per follow_prompt_id."""
        latest: dict[int | None, Event] = {}
        for event in events:
            key = event.follow_prompt_id
            if key not in latest or event.occurred_at > latest[key].occurred_at:
                latest[key] = event
        return sorted(latest.values(), key=lambda e: e.occurred_at, reverse=True)

    def _load_follow_topics(self, sender: str) -> dict[int, str]:
        """Load follow prompt ID → topic text mapping for a user."""
        follows = self.db.follow_prompts.get_active(sender)
        return {fp.id: fp.prompt_text for fp in follows if fp.id is not None}

    @staticmethod
    def _format_event(event: Event, follow_topics: dict[int, str]) -> str:
        """Format a single event for context injection."""
        date = event.occurred_at.strftime("%Y-%m-%d")
        topic = follow_topics.get(event.follow_prompt_id, "") if event.follow_prompt_id else ""
        topic_label = f" [{topic}]" if topic else ""
        line = f"I saw in the news [{date}]{topic_label}: {event.headline}"
        if event.source_url:
            line += f"\n{event.source_url}"
        return line

    # ── Entity context retrieval ──────────────────────────────────────────

    async def _retrieve_entity_messages(
        self, content: str, sender: str
    ) -> list[tuple[str, str]] | None:
        """Embed message, find similar entities, return individual (role, content) per entity."""
        if not self._embedding_model_client:
            return None

        candidates, entities = self._load_entity_candidates(sender)
        if not candidates:
            return None

        query_vec = await embed_text(self._embedding_model_client, content)
        if query_vec is None:
            return None

        matches = find_similar(
            query_vec,
            candidates,
            top_k=int(self.config.runtime.ENTITY_CONTEXT_TOP_K),
            threshold=self.config.runtime.ENTITY_CONTEXT_THRESHOLD,
        )
        if not matches:
            return None

        messages, total_facts = self._build_entity_messages(matches, entities)
        if messages:
            logger.info(
                "Entity context: %d matches, %d facts, top_score=%.2f",
                len(matches),
                total_facts,
                matches[0][1],
            )
        return messages or None

    def _load_entity_candidates(
        self, sender: str
    ) -> tuple[list[tuple[int, list[float]]], list[Entity]]:
        """Load user entities with embeddings as (id, vector) pairs."""
        entities = self.db.entities.get_for_user(sender)
        candidates: list[tuple[int, list[float]]] = []
        for entity in entities:
            if entity.embedding is None or entity.id is None:
                continue
            candidates.append((entity.id, deserialize_embedding(entity.embedding)))
        return candidates, entities

    def _build_entity_messages(
        self,
        matches: list[tuple[int, float]],
        entities: list[Entity],
    ) -> tuple[list[tuple[str, str]], int]:
        """Build individual (role, content) messages per matched entity."""
        entity_map = {e.id: e for e in entities if e.id is not None}
        messages: list[tuple[str, str]] = []
        total_facts = 0

        for entity_id, _score in matches:
            entity = entity_map.get(entity_id)
            if not entity or entity.id is None:
                continue

            facts = self.db.facts.get_for_entity(entity.id)
            if not facts:
                continue

            fact_texts = [
                f.content for f in facts[: int(self.config.runtime.ENTITY_CONTEXT_MAX_FACTS)]
            ]
            label = f"{entity.name} ({entity.tagline})" if entity.tagline else entity.name
            content = f"I know about {label}: {'; '.join(fact_texts)}"
            messages.append((MessageRole.SYSTEM.value, content))
            total_facts += len(fact_texts)

        return messages, total_facts

    # ── Per-step hooks ─────────────────────────────────────────────────────

    async def _after_step(self, step_records: list[ToolCallRecord], messages: list[dict]) -> None:
        """Persist reasoning as thoughts and inject entity context for next step."""
        self._persist_step_reasoning(step_records)
        await self._inject_reasoning_entity_context(step_records, messages)

    def _should_stop_loop(self, step_records: list[ToolCallRecord]) -> bool:
        """Stop the agentic loop after message_user is called."""
        return any(r.tool == MessageUserTool.name for r in step_records)

    def _persist_step_reasoning(self, step_records: list[ToolCallRecord]) -> None:
        """Write reasoning from this step's tool calls to the thought log."""
        if not self._current_user:
            return
        for record in step_records:
            if not record.reasoning:
                continue
            args_summary = self._summarize_args(record)
            thought = f"[{record.tool}({args_summary})] {record.reasoning}"
            self.db.thoughts.add(self._current_user, thought)
            logger.info("[thought] %s", thought[:200])

    @staticmethod
    def _summarize_args(record: ToolCallRecord) -> str:
        """One-line summary of tool arguments for thought log."""
        if not record.arguments:
            return ""
        for v in record.arguments.values():
            if isinstance(v, str):
                return v[:60]
        return str(next(iter(record.arguments.values())))[:60]

    async def _inject_reasoning_entity_context(
        self, step_records: list[ToolCallRecord], messages: list[dict]
    ) -> None:
        """Embed reasoning from this step, find similar entities, inject as context."""
        if not self._current_user or not self._embedding_model_client:
            return
        reasoning_texts = [r.reasoning for r in step_records if r.reasoning]
        if not reasoning_texts:
            return
        anchor = " ".join(reasoning_texts)
        entity_messages = await self._retrieve_entity_messages(anchor, self._current_user)
        if entity_messages:
            for role, content in entity_messages:
                messages.append({"role": role, "content": content})
            logger.debug(
                "Injected mid-step entity context from reasoning (%d entities)",
                len(entity_messages),
            )
