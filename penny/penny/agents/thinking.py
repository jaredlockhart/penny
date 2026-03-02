"""ThinkingAgent — Penny's autonomous thinking layer.

Runs on the scheduler with high priority. Each cycle:
1. Builds shared context (profile, messages, interests, thoughts)
2. Builds per-user tools (think, recall, search, fetch_news, message_user)
3. Runs an agentic loop where the model decides what to do
4. Thoughts get persisted; messages get sent; or nothing happens (quiet cycle)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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

    Runs an agentic loop with tools for thinking, recalling memory,
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
        """One thinking cycle: build context, build tools, run agentic loop."""
        history = await self._build_context(user)
        tools = self._build_tools(user)
        if not tools:
            return

        self._install_tools(tools)

        try:
            logger.info("Inner monologue cycle starting for %s", user)
            max_steps = int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)
            await self.run(
                prompt="Begin your thinking cycle.",
                history=history,
                use_tools=True,
                max_steps=max_steps,
                system_prompt=Prompt.INNER_MONOLOGUE_SYSTEM_PROMPT,
            )
            logger.info("Inner monologue cycle complete for %s", user)
        except Exception:
            logger.exception("Inner monologue cycle failed for %s", user)

    def _build_tools(self, user: str) -> list[Tool]:
        """Extend base tools with MessageUserTool when channel is available."""
        tools = super()._build_tools(user)
        if self._channel:
            tools.append(MessageUserTool(channel=self._channel, user=user))
        return tools
