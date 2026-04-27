"""NotifyAgent — Penny's proactive outreach.

Runs on the scheduler when the user has unnotified thoughts and the
cooldown has elapsed.  Each cycle is a fully model-driven agent
loop: the system prompt steers the model through reading its
unnotified thoughts, picking one to share, moving it to the
notified-thoughts collection, sending the message via
``send_message``, and exiting via ``done()``.  Python wraps the
cycle with eligibility guardrails (channel set, user not muted,
cooldown elapsed) — everything else is handled by tools.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from penny.agents.base import Agent
from penny.constants import NotifyPromptType, PennyConstants
from penny.prompts import Prompt
from penny.tools.send_message import SendMessageTool

logger = logging.getLogger(__name__)


class NotifyAgent(Agent):
    """Background outreach agent — sends thoughts when the user is idle."""

    name = "notify"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._boot_time = datetime.now(UTC).replace(tzinfo=None)
        # Tools stay available on the final agentic step — notify exits
        # via ``done()`` once it has called ``send_message``.
        self._keep_tools_on_final_step = True

    def get_max_steps(self) -> int:
        """Read from config so /config changes take effect immediately."""
        return int(self.config.runtime.MESSAGE_MAX_STEPS)

    # ── Scheduled entry point ────────────────────────────────────────────

    async def execute_for_user(self, user: str) -> bool:
        """Run a notify cycle if the user is eligible."""
        if not self._should_notify(user):
            return False
        return await self._run_notify_cycle(user)

    # ── Eligibility ──────────────────────────────────────────────────────

    def _should_notify(self, user: str) -> bool:
        """Python-space eligibility checks."""
        if not self._channel:
            return False
        if self.db.users.is_muted(user):
            return False
        if not self._has_unnotified_thoughts():
            return False
        return self._cooldown_elapsed(user)

    def _has_unnotified_thoughts(self) -> bool:
        """Any pending thought in the unnotified-thoughts collection."""
        return len(self.db.memories.read_latest(PennyConstants.MEMORY_UNNOTIFIED_THOUGHTS, k=1)) > 0

    def _cooldown_elapsed(self, user: str) -> bool:
        """Exponential backoff cooldown between autonomous outreach messages."""
        latest = self.db.messages.get_latest_autonomous_outgoing_time(user)
        if latest is None:
            return True
        elapsed = (datetime.now(UTC).replace(tzinfo=None) - latest).total_seconds()
        count = self.db.messages.count_autonomous_since_last_incoming(user, self._boot_time)
        cooldown = min(
            self.config.runtime.NOTIFY_COOLDOWN_MIN * (2 ** max(count - 1, 0)),
            self.config.runtime.NOTIFY_COOLDOWN_MAX,
        )
        return elapsed >= cooldown

    # ── Cycle ────────────────────────────────────────────────────────────

    async def _run_notify_cycle(self, user: str) -> bool:
        """Run one model-driven notify cycle.

        The cycle succeeds when the model calls ``send_message`` at
        least once before exiting; otherwise the cycle is treated as a
        skip (model decided nothing was worth sharing).
        """
        self._install_tools(self.get_tools(user))
        run_id = uuid.uuid4().hex
        response = await self.run(
            prompt="",
            max_steps=self.get_max_steps(),
            system_prompt=Prompt.NOTIFY_SYSTEM_PROMPT,
            run_id=run_id,
            prompt_type=NotifyPromptType.CYCLE,
        )
        return any(record.tool == SendMessageTool.name for record in response.tool_calls)
