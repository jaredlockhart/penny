"""ThinkingAgent — Penny's autonomous thinking layer.

Runs on the scheduler after extraction. Each cycle:
1. Builds full context (profile, messages, interests, events, thoughts)
2. Orientation step: model reads full context, writes a focused plan (no tools)
3. Agentic loop: lean context (profile + plan + entities) with tools
4. Reasoning from tool calls gets persisted as thoughts
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.agents.models import MessageRole
from penny.agents.penny_agent import PennyAgent
from penny.prompts import Prompt
from penny.tools.base import Tool
from penny.tools.message_user import MessageUserTool

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel
    from penny.tools.news import NewsTool

logger = logging.getLogger(__name__)


class ThinkingAgent(PennyAgent):
    """Autonomous inner monologue — Penny's conscious mind.

    Runs an agentic loop with tools for recalling memory,
    searching the web, fetching news, and messaging the user.
    """

    THOUGHT_CONTEXT_LIMIT = 50

    def __init__(
        self,
        search_tool: Tool | None = None,
        news_tool: NewsTool | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(search_tool=search_tool, news_tool=news_tool, **kwargs)
        self._channel: MessageChannel | None = None

    @property
    def name(self) -> str:
        return "inner_monologue"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending messages to users."""
        self._channel = channel

    async def execute(self) -> bool:
        """Run one inner monologue cycle for each user."""
        users = self.db.users.get_all_senders()
        if not users:
            return False

        for user in users:
            if self.db.users.is_muted(user):
                continue
            await self._run_cycle(user)

        return True

    async def _run_cycle(self, user: str) -> None:
        """One thinking cycle — summary method.

        Orientation reads full context, agentic loop gets lean context.
        """
        self._current_user = user
        full_history = await self._build_context(user)
        tools = self._build_tools(user)
        if not tools:
            self._current_user = None
            return

        self._install_tools(tools)

        try:
            logger.info("Inner monologue cycle starting for %s", user)
            orientation = await self._orient(full_history)
            loop_history = self._build_loop_context(user, orientation)
            loop_history = await self._inject_entity_context(
                orientation,
                user,
                loop_history,
            )
            max_steps = int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)
            await self.run(
                prompt=Prompt.INNER_MONOLOGUE_BEGIN_PROMPT,
                history=loop_history,
                use_tools=True,
                max_steps=max_steps,
                system_prompt=Prompt.INNER_MONOLOGUE_SYSTEM_PROMPT,
            )
            logger.info("Inner monologue cycle complete for %s", user)
        except Exception:
            logger.exception("Inner monologue cycle failed for %s", user)
        finally:
            self._current_user = None

    # ── Context ────────────────────────────────────────────────────────────

    def _build_loop_context(
        self,
        user: str,
        orientation: str,
    ) -> list[tuple[str, str]]:
        """Build lean context for the agentic loop: profile + plan only."""
        history = self._inject_profile_context(user, None, None)
        history = history or []
        history.append((MessageRole.SYSTEM.value, f"## Your Plan\n{orientation}"))
        return history

    # ── Orientation ────────────────────────────────────────────────────────

    async def _orient(self, history: list[tuple[str, str]] | None) -> str:
        """Orientation step — model reads all context, emits planning text."""
        response = await self._compose_user_facing(
            prompt=Prompt.ORIENTATION_PROMPT,
            history=history,
            system_prompt=Prompt.INNER_MONOLOGUE_SYSTEM_PROMPT,
        )
        thought = response.answer or ""
        if thought and self._current_user:
            self.db.thoughts.add(self._current_user, f"[orientation] {thought}")
            logger.info("[orientation] %s", thought[:200])
        return thought

    # ── Tools ──────────────────────────────────────────────────────────────

    def _build_tools(self, user: str) -> list[Tool]:
        """Extend base tools with MessageUserTool when channel is available."""
        tools = super()._build_tools(user)
        if self._channel:
            tools.append(MessageUserTool(channel=self._channel, user=user))
        return tools
