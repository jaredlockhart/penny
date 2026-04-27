"""NotifyAgent — Penny's proactive outreach.

Runs on the scheduler when the user has unnotified thoughts and
the cooldown has elapsed.  Each cycle is a fully model-driven
agent loop: the system prompt steers the model through reading
its unnotified thoughts, picking one to share, moving it to the
notified-thoughts collection, and producing the message text as
its final answer.  Python wraps the cycle with eligibility
guardrails (channel set, user not muted, cooldown elapsed) and
sends the answer through the channel.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from penny.agents.base import Agent
from penny.constants import NotifyPromptType, PennyConstants
from penny.prompts import Prompt

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class NotifyAgent(Agent):
    """Background outreach agent — sends thoughts when the user is idle."""

    name = "notify"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._boot_time = datetime.now(UTC).replace(tzinfo=None)
        self._channel: MessageChannel | None = None

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel used to deliver notifications."""
        self._channel = channel

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
        """Run one model-driven notify cycle and deliver its answer."""
        channel = self._channel
        if channel is None:
            raise RuntimeError("notify cycle started without a channel set")
        self._install_tools(self.get_tools(user))
        run_id = uuid.uuid4().hex
        response = await self.run(
            prompt="",
            max_steps=self.get_max_steps(),
            system_prompt=Prompt.NOTIFY_SYSTEM_PROMPT,
            run_id=run_id,
            prompt_type=NotifyPromptType.CYCLE,
        )
        answer = response.answer.strip() if response.answer else ""
        if not answer:
            logger.info("Notify cycle produced no message for %s", user)
            return False
        await channel.send_response(
            user, answer, parent_id=None, author=self.name, quote_message=None
        )
        logger.info("Notification sent to %s", user)
        return True
