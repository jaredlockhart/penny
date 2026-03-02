"""PennyAgent — shared base for Penny's conversation and thinking agents.

Provides common tool building, context injection, interest profiles, and
entity/event retrieval that both ChatAgent and ThinkingAgent use.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.agents.models import MessageRole
from penny.commands.models import CommandContext
from penny.constants import PennyConstants
from penny.database.models import Entity, Event
from penny.ollama.embeddings import deserialize_embedding, find_similar
from penny.ollama.similarity import embed_text
from penny.tools.base import Tool, ToolExecutor, ToolRegistry
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.follow import FollowTool
from penny.tools.learn import LearnTool
from penny.tools.recall import RecallTool
from penny.tools.search import SearchTool
from penny.tools.think import ThinkTool

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

    # ── Tool building ──────────────────────────────────────────────────────

    def _build_tools(self, user: str) -> list[Tool]:
        """Build the shared tool list. Subclasses can extend via override."""
        searches = int(self.config.runtime.LEARN_PROMPT_DEFAULT_SEARCHES)
        tools: list[Tool] = [
            ThinkTool(db=self.db, user=user),
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
        """Build shared context for both conversation and thinking modes.

        When content is provided (conversation mode), also injects
        embedding-similar entity and event context using recent
        conversation as the embedding anchor.
        """
        history = self._inject_profile_context(user, None, content)
        history = self._inject_message_context(user, history)
        history = self._inject_interest_context(user, history)
        if content:
            query = self._build_embedding_query(content, user)
            history = await self._inject_entity_context(query, user, history)
            history = await self._inject_event_context(query, user, history)
        history = self._inject_thought_context(user, history)
        return history

    def _build_embedding_query(self, content: str, user: str) -> str:
        """Build embedding query from current message + recent conversation."""
        try:
            limit = int(self.config.runtime.MESSAGE_CONTEXT_LIMIT)
            messages = self.db.messages.get_conversation(user, limit=limit)
            if not messages:
                return content
            recent = "\n".join(msg.content for msg in messages)
            return f"{content}\n{recent}"
        except Exception:
            return content

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

            profile_summary = f"The user's name is {user_info.name}."
            history = history or []
            history = [(MessageRole.SYSTEM.value, profile_summary), *history]
            logger.debug("Injected profile context for %s", sender)

            if content is not None:
                name = user_info.name
                user_said_name = bool(re.search(rf"\b{re.escape(name)}\b", content, re.IGNORECASE))
                if self._search_tool and isinstance(self._search_tool, SearchTool):
                    self._search_tool.redact_terms = [] if user_said_name else [name]
        except Exception:
            pass

        return history

    def _inject_message_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject recent conversation messages as context."""
        try:
            limit = int(self.config.runtime.MESSAGE_CONTEXT_LIMIT)
            messages = self.db.messages.get_conversation(sender, limit=limit)
            if not messages:
                return history

            lines = []
            for msg in messages:
                ts = msg.timestamp.strftime("%H:%M")
                direction = (
                    "User" if msg.direction == PennyConstants.MessageDirection.INCOMING else "Penny"
                )
                lines.append(f"[{ts}] {direction}: {msg.content}")

            context = "Recent conversation:\n" + "\n".join(lines)
            history = history or []
            history = [*history, (MessageRole.SYSTEM.value, context)]
            logger.debug("Injected message context (%d messages)", len(messages))
        except Exception:
            logger.warning("Message context retrieval failed, proceeding without")
        return history

    def _inject_interest_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject a broad interest profile from entities, learn topics, and follows."""
        try:
            context = self._build_interest_profile(sender)
            if not context:
                return history
            history = history or []
            history = [*history, (MessageRole.SYSTEM.value, context)]
            logger.debug("Injected interest context")
        except Exception:
            logger.warning("Interest context retrieval failed, proceeding without")
        return history

    async def _inject_entity_context(
        self,
        content: str,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Retrieve entity knowledge and append it to history."""
        try:
            entity_context = await self._retrieve_entity_context(content, sender)
            if entity_context:
                history = history or []
                history = [*history, (MessageRole.SYSTEM.value, entity_context)]
                logger.debug("Injected entity context (%d chars)", len(entity_context))
        except Exception:
            logger.warning("Entity context retrieval failed, proceeding without")
        return history

    async def _inject_event_context(
        self,
        content: str,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Embed message and find similar events to inject as context."""
        try:
            event_context = await self._retrieve_event_context(content, sender)
            if event_context:
                history = history or []
                history = [*history, (MessageRole.SYSTEM.value, event_context)]
                logger.debug("Injected event context (%d chars)", len(event_context))
        except Exception:
            logger.warning("Event context retrieval failed, proceeding without")
        return history

    def _inject_thought_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]] | None:
        """Inject recent inner monologue thoughts as context."""
        try:
            thoughts = self.db.thoughts.get_recent(sender, limit=self.THOUGHT_CONTEXT_LIMIT)
            if not thoughts:
                return history
            thought_text = "Your recent thoughts:\n" + "\n".join(
                f"- [{t.created_at.strftime('%Y-%m-%d %H:%M')}] {t.content}" for t in thoughts
            )
            history = history or []
            history = [*history, (MessageRole.SYSTEM.value, thought_text)]
            logger.debug("Injected thought context (%d thoughts)", len(thoughts))
        except Exception:
            logger.warning("Thought context retrieval failed, proceeding without")
        return history

    # ── Interest profile ───────────────────────────────────────────────────

    def _build_interest_profile(self, user: str) -> str | None:
        """Build a compact summary of the user's known interests."""
        sections: list[str] = []
        self._add_entity_interests(user, sections)
        self._add_learn_interests(user, sections)
        self._add_follow_interests(user, sections)
        if not sections:
            return None
        return "User's known interests:\n" + "\n".join(sections)

    def _add_entity_interests(self, user: str, sections: list[str]) -> None:
        """Add top entities by recency to interest sections."""
        entities = self.db.entities.get_for_user(user)
        if not entities:
            return
        sorted_ents = sorted(entities, key=lambda e: e.created_at, reverse=True)[:30]
        names = [f"{e.name} ({e.tagline})" if e.tagline else e.name for e in sorted_ents]
        sections.append("Topics: " + ", ".join(names))

    def _add_learn_interests(self, user: str, sections: list[str]) -> None:
        """Add past learn topics to interest sections."""
        prompts = self.db.learn_prompts.get_for_user(user)
        if not prompts:
            return
        topics = [lp.prompt_text for lp in prompts]
        sections.append("Topics the user asked you to research: " + ", ".join(topics))

    def _add_follow_interests(self, user: str, sections: list[str]) -> None:
        """Add active follow subscriptions to interest sections."""
        follows = self.db.follow_prompts.get_active(user)
        if not follows:
            return
        topics = [fp.prompt_text for fp in follows]
        sections.append("Topics the user has you actively following: " + ", ".join(topics))

    # ── Entity context retrieval ──────────────────────────────────────────

    async def _retrieve_entity_context(self, content: str, sender: str) -> str | None:
        """Embed message, find similar entities, format their facts as context."""
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

        context_text, total_facts = self._build_entity_context_text(matches, entities)
        if context_text:
            logger.info(
                "Entity context: %d matches, %d facts, top_score=%.2f",
                len(matches),
                total_facts,
                matches[0][1],
            )
        return context_text

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

    def _build_entity_context_text(
        self,
        matches: list[tuple[int, float]],
        entities: list[Entity],
    ) -> tuple[str | None, int]:
        """Build formatted context text from matched entities and their facts."""
        entity_map = {e.id: e for e in entities if e.id is not None}
        context_lines: list[str] = []
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
            context_lines.append(f"- {label}: {'; '.join(fact_texts)}")
            total_facts += len(fact_texts)

        if not context_lines:
            return None, 0

        context_text = "Relevant knowledge:\n" + "\n".join(context_lines)
        return context_text, total_facts

    # ── Event context retrieval ───────────────────────────────────────────

    async def _retrieve_event_context(self, content: str, sender: str) -> str | None:
        """Embed message, find similar events, format as context."""
        if not self._embedding_model_client:
            return None

        candidates = self._load_event_candidates(sender)
        if not candidates:
            return None

        query_vec = await embed_text(self._embedding_model_client, content)
        if query_vec is None:
            return None

        matches = find_similar(
            query_vec,
            candidates,
            top_k=int(self.config.runtime.EVENT_CONTEXT_TOP_K),
            threshold=self.config.runtime.EVENT_CONTEXT_THRESHOLD,
        )
        if not matches:
            return None

        events = self.db.events.get_for_user(sender)
        context_text = self._build_event_context_text(matches, events)
        if context_text:
            logger.info(
                "Event context: %d matches, top_score=%.2f",
                len(matches),
                matches[0][1],
            )
        return context_text

    def _load_event_candidates(self, sender: str) -> list[tuple[int, list[float]]]:
        """Load user events with embeddings as (id, vector) pairs."""
        events = self.db.events.get_for_user(sender)
        candidates: list[tuple[int, list[float]]] = []
        for event in events:
            if event.embedding is None or event.id is None:
                continue
            candidates.append((event.id, deserialize_embedding(event.embedding)))
        return candidates

    def _build_event_context_text(
        self,
        matches: list[tuple[int, float]],
        events: list[Event],
    ) -> str | None:
        """Build formatted context text from matched events."""
        event_map = {e.id: e for e in events if e.id is not None}
        context_lines: list[str] = []

        for event_id, _score in matches:
            event = event_map.get(event_id)
            if not event:
                continue
            date = event.occurred_at.strftime("%Y-%m-%d")
            line = f"- [{date}] {event.headline}"
            if event.summary:
                line += f": {event.summary[:200]}"
            if event.source_url:
                line += f" ({event.source_url})"
            context_lines.append(line)

        if not context_lines:
            return None

        return "Recent relevant events:\n" + "\n".join(context_lines)
