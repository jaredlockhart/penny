"""SendMessageTool — model-driven outbound message delivery.

Bound at construction to a specific (channel, recipient, agent_name)
triple plus the database and runtime config.  The model calls this
tool with a message body when it has decided what to say.  The
tool checks two gates before dispatching:

- **Mute**: if the recipient has muted notifications, the tool
  refuses with a string that tells the model to call ``done()``.
- **Cooldown**: exponential backoff between autonomous sends from
  the same agent.  Counts entries in ``penny-messages`` authored by
  this agent since the recipient's last entry in ``user-messages``;
  cooldown doubles per backoff step, capped at the configured max.

Both gates apply to all callers — chat agents that reply via the
final-answer mechanism never invoke this tool, so the gates are
effectively notify-only in practice.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from penny.constants import PennyConstants
from penny.tools.base import Tool
from penny.tools.models import SendMessageArgs

if TYPE_CHECKING:
    from penny.channels.base import MessageChannel
    from penny.config import Config
    from penny.database import Database

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Send a message to the user through the bound channel."""

    name = "send_message"
    description = (
        "Send a message to the user.  Use this once you have decided "
        "what to say — the ``content`` is the exact text the user will "
        "see.  The send is gated on mute state and an exponential "
        "backoff cooldown; if either refuses, the response will say so "
        "and you should call ``done`` to exit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text to send to the user.",
            }
        },
        "required": ["content"],
    }

    _MUTED_RESPONSE = (
        "Message NOT sent: the user has muted autonomous messages.  "
        "Call ``done`` to exit — do not retry."
    )
    _COOLDOWN_RESPONSE = (
        "Message NOT sent: cooldown has not elapsed since the last "
        "autonomous send.  Call ``done`` to exit — do not retry."
    )

    def __init__(
        self,
        channel: MessageChannel,
        recipient: str,
        agent_name: str,
        db: Database,
        config: Config,
    ) -> None:
        self._channel = channel
        self._recipient = recipient
        self._agent_name = agent_name
        self._db = db
        self._config = config

    async def execute(self, **kwargs: Any) -> str:
        args = SendMessageArgs(**kwargs)
        if self._is_muted():
            logger.info("send_message refused (muted): %s", self._recipient)
            return self._MUTED_RESPONSE
        if not self._cooldown_elapsed():
            logger.info(
                "send_message refused (cooldown): %s → %s", self._agent_name, self._recipient
            )
            return self._COOLDOWN_RESPONSE
        await self._channel.send_response(
            recipient=self._recipient,
            content=args.content,
            parent_id=None,
            author=self._agent_name,
            quote_message=None,
        )
        logger.info("send_message: %s → %s", self._agent_name, self._recipient)
        return "Message sent."

    # ── Gating helpers ──────────────────────────────────────────────────

    def _is_muted(self) -> bool:
        return self._db.users.is_muted(self._recipient)

    def _cooldown_elapsed(self) -> bool:
        """Exponential backoff: cooldown doubles per send since the last user message."""
        latest = self._latest_send_time()
        if latest is None:
            return True
        elapsed = (_naive_utc_now() - _to_naive(latest)).total_seconds()
        count = self._count_sends_since_user_message()
        cooldown = min(
            self._config.runtime.NOTIFY_COOLDOWN_MIN * (2 ** max(count - 1, 0)),
            self._config.runtime.NOTIFY_COOLDOWN_MAX,
        )
        return elapsed >= cooldown

    def _latest_send_time(self) -> datetime | None:
        """Created-at of this agent's most recent ``penny-messages`` entry."""
        for entry in self._db.memories.read_latest(PennyConstants.MEMORY_PENNY_MESSAGES_LOG):
            if entry.author == self._agent_name:
                return entry.created_at
        return None

    def _count_sends_since_user_message(self) -> int:
        """Number of this agent's sends newer than the latest ``user-messages`` entry."""
        latest_user = self._latest_user_message_time()
        cutoff = _to_naive(latest_user) if latest_user is not None else None
        count = 0
        for entry in self._db.memories.read_latest(PennyConstants.MEMORY_PENNY_MESSAGES_LOG):
            if entry.author != self._agent_name:
                continue
            if cutoff is not None and _to_naive(entry.created_at) <= cutoff:
                break
            count += 1
        return count

    def _latest_user_message_time(self) -> datetime | None:
        """Created-at of the most recent ``user-messages`` entry."""
        entries = self._db.memories.read_latest(PennyConstants.MEMORY_USER_MESSAGES_LOG, k=1)
        return entries[0].created_at if entries else None


def _naive_utc_now() -> datetime:
    """Naive UTC ``now`` to compare against ``MemoryEntry.created_at``,
    which round-trips through SQLite as a tz-naive value."""
    return datetime.now(UTC).replace(tzinfo=None)


def _to_naive(value: datetime) -> datetime:
    """Strip tzinfo if present so naive/aware mixes don't crash arithmetic."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
