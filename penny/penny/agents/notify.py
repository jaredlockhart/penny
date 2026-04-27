"""NotifyAgent — Penny's proactive outreach.

Runs on the scheduler when the user has unnotified thoughts and the
cooldown has elapsed.  Each cycle is a fully model-driven agent
loop: the prompt steers the model through reading its unnotified
thoughts, picking one to share, moving it to ``notified-thoughts``,
sending the message via ``send_message``, and exiting.  Python
wraps the cycle with eligibility guardrails (channel set, user not
muted, cooldown elapsed) — everything else is handled by tools.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from penny.agents.base import Agent
from penny.constants import NotifyPromptType, PennyConstants
from penny.tools.send_message import SendMessageTool


class NotifyAgent(Agent):
    """Background outreach agent — sends thoughts when the user is idle."""

    name = "notify"
    prompt_type = NotifyPromptType.CYCLE
    # The cycle ends when the model has called ``send_message`` to deliver
    # the notification — that's the success signal, not ``done``.
    terminator_tool = SendMessageTool.name

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._boot_time = datetime.now(UTC)

    # ── Scheduled entry point ────────────────────────────────────────────

    async def execute_for_user(self, user: str) -> bool:
        """Run a notify cycle if the user is eligible."""
        if not self._should_notify(user):
            return False
        return await self._run_cycle(user)

    # ── Eligibility ──────────────────────────────────────────────────────

    def _should_notify(self, user: str) -> bool:
        """Python-space eligibility checks."""
        if not self._channel:
            return False
        if self.db.users.is_muted(user):
            return False
        if not self._has_unnotified_thoughts():
            return False
        return self._cooldown_elapsed()

    def _has_unnotified_thoughts(self) -> bool:
        """Any pending thought in the unnotified-thoughts collection."""
        return len(self.db.memories.read_latest(PennyConstants.MEMORY_UNNOTIFIED_THOUGHTS, k=1)) > 0

    def _cooldown_elapsed(self) -> bool:
        """Exponential backoff cooldown between autonomous outreach messages.

        Reads the ``penny-messages`` log filtered by ``author == self.name``
        to find prior notifications and ``user-messages`` for the user's
        most recent reply.
        """
        latest = self._latest_notify_time()
        if latest is None:
            return True
        elapsed = (datetime.now(UTC) - latest).total_seconds()
        count = self._count_notifies_since_user_response()
        cooldown = min(
            self.config.runtime.NOTIFY_COOLDOWN_MIN * (2 ** max(count - 1, 0)),
            self.config.runtime.NOTIFY_COOLDOWN_MAX,
        )
        return elapsed >= cooldown

    def _latest_notify_time(self) -> datetime | None:
        """Created-at of the most recent ``penny-messages`` entry from notify."""
        for entry in self.db.memories.read_latest(PennyConstants.MEMORY_PENNY_MESSAGES_LOG):
            if entry.author == self.name:
                return entry.created_at
        return None

    def _count_notifies_since_user_response(self) -> int:
        """Count notify-authored ``penny-messages`` entries newer than the cutoff.

        Cutoff is the later of ``self._boot_time`` and the user's latest
        ``user-messages`` entry — so a service restart resets the
        backoff count, and so does a fresh user message.
        """
        latest_user = self._latest_user_message_time()
        cutoff = self._boot_time
        if latest_user is not None and latest_user > cutoff:
            cutoff = latest_user
        count = 0
        for entry in self.db.memories.read_latest(PennyConstants.MEMORY_PENNY_MESSAGES_LOG):
            if entry.author != self.name:
                continue
            if entry.created_at <= cutoff:
                break
            count += 1
        return count

    def _latest_user_message_time(self) -> datetime | None:
        """Created-at of the most recent ``user-messages`` entry."""
        entries = self.db.memories.read_latest(PennyConstants.MEMORY_USER_MESSAGES_LOG, k=1)
        return entries[0].created_at if entries else None
