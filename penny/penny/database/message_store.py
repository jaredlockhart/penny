"""Message store — logging, threading, and queries for messages."""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import Session, func, select

from penny.agents.models import MessageRole
from penny.constants import PennyConstants
from penny.database.models import CommandLog, MessageLog, PromptLog

logger = logging.getLogger(__name__)

# Patterns for stripping markdown formatting from outgoing messages
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_STRIKETHROUGH_RE = re.compile(r"~{1,2}(.+?)~{1,2}")
_MONOSPACE_RE = re.compile(r"`(.+?)`")
_TILDE_OPERATOR = "\u223c"


class MessageStore:
    """Manages MessageLog, PromptLog, and CommandLog records."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    @staticmethod
    def strip_formatting(text: str) -> str:
        """Strip markdown formatting for quote lookup.

        Signal converts **bold**/etc. to native formatting, so quotes come back
        as plain text. We strip these markers to enable reliable matching.
        """
        text = _BOLD_ITALIC_RE.sub(r"\1", text)
        text = _STRIKETHROUGH_RE.sub(r"\1", text)
        text = _MONOSPACE_RE.sub(r"\1", text)
        text = text.replace(_TILDE_OPERATOR, "~")
        return text

    # --- Message logging ---

    def log_message(
        self,
        direction: str,
        sender: str,
        content: str,
        parent_id: int | None = None,
        signal_timestamp: int | None = None,
        external_id: str | None = None,
        is_reaction: bool = False,
        recipient: str | None = None,
    ) -> int | None:
        """Log a user message or agent response. Returns the message ID or None."""
        if direction == PennyConstants.MessageDirection.OUTGOING:
            content = self.strip_formatting(content)
        try:
            with self._session() as session:
                log = MessageLog(
                    direction=direction,
                    sender=sender,
                    content=content,
                    parent_id=parent_id,
                    signal_timestamp=signal_timestamp,
                    external_id=external_id,
                    is_reaction=is_reaction,
                    recipient=recipient,
                )
                session.add(log)
                session.commit()
                session.refresh(log)
                logger.debug("Logged %s message from %s (id=%d)", direction, sender, log.id)
                return log.id
        except Exception as e:
            logger.error("Failed to log message: %s", e)
            return None

    def log_prompt(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        tools: list[dict] | None = None,
        thinking: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Log a prompt/response exchange with Ollama."""
        import json

        try:
            with self._session() as session:
                log = PromptLog(
                    model=model,
                    messages=json.dumps(messages),
                    tools=json.dumps(tools) if tools else None,
                    response=json.dumps(response),
                    thinking=thinking,
                    duration_ms=duration_ms,
                )
                session.add(log)
                session.commit()
                logger.debug("Logged prompt exchange (model=%s)", model)
        except Exception as e:
            logger.error("Failed to log prompt: %s", e)

    def log_command(
        self,
        user: str,
        channel_type: str,
        command_name: str,
        command_args: str,
        response: str,
        error: str | None = None,
    ) -> None:
        """Log a command invocation."""
        try:
            with self._session() as session:
                log = CommandLog(
                    user=user,
                    channel_type=channel_type,
                    command_name=command_name,
                    command_args=command_args,
                    response=response,
                    error=error,
                )
                session.add(log)
                session.commit()
                logger.debug("Logged command: /%s %s", command_name, command_args)
        except Exception as e:
            logger.error("Failed to log command: %s", e)

    # --- Message metadata ---

    def set_signal_timestamp(self, message_id: int, signal_timestamp: int) -> None:
        """Update the Signal timestamp on a message after sending."""
        try:
            with self._session() as session:
                msg = session.get(MessageLog, message_id)
                if msg:
                    msg.signal_timestamp = signal_timestamp
                    session.add(msg)
                    session.commit()
        except Exception as e:
            logger.error("Failed to set signal_timestamp: %s", e)

    def set_external_id(self, message_id: int, external_id: str) -> None:
        """Update the external ID on a message after sending."""
        try:
            with self._session() as session:
                msg = session.get(MessageLog, message_id)
                if msg:
                    msg.external_id = external_id
                    session.add(msg)
                    session.commit()
        except Exception as e:
            logger.error("Failed to set external_id: %s", e)

    # --- Message lookup ---

    def get_by_id(self, message_id: int) -> MessageLog | None:
        """Get a message by its database ID."""
        with self._session() as session:
            return session.get(MessageLog, message_id)

    def find_by_external_id(self, external_id: str) -> MessageLog | None:
        """Find a message by its platform-specific external ID."""
        with self._session() as session:
            return session.exec(
                select(MessageLog).where(MessageLog.external_id == external_id)
            ).first()

    def find_outgoing_by_content(self, content: str) -> MessageLog | None:
        """Find the most recent outgoing message matching the given content prefix."""
        content = self.strip_formatting(content)
        with self._session() as session:
            return session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.content.startswith(content),
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
            ).first()

    # --- Thread context ---

    def get_thread_context(
        self, quoted_text: str
    ) -> tuple[int | None, list[tuple[str, str]] | None]:
        """Look up a quoted message and return its id and conversation context."""
        parent_msg = self.find_outgoing_by_content(quoted_text)
        if not parent_msg:
            logger.warning("Could not find quoted message in database")
            return None, None

        assert parent_msg.id is not None
        thread = self._walk_thread(parent_msg.id)
        history = [
            (
                MessageRole.USER
                if m.direction == PennyConstants.MessageDirection.INCOMING
                else MessageRole.ASSISTANT,
                m.content,
            )
            for m in thread
        ]
        logger.info("Built thread history with %d messages", len(history))
        return parent_msg.id, history

    def _walk_thread(self, message_id: int, limit: int = 20) -> list[MessageLog]:
        """Walk up the parent chain. Returns messages oldest-first."""
        history: list[MessageLog] = []
        with self._session() as session:
            current_id: int | None = message_id
            while current_id is not None and len(history) < limit:
                msg = session.get(MessageLog, current_id)
                if msg is None:
                    break
                history.append(msg)
                current_id = msg.parent_id
        history.reverse()
        return history

    # --- Conversation queries ---

    def get_conversation_leaves(self) -> list[MessageLog]:
        """Get outgoing leaf messages eligible for spontaneous continuation."""
        with self._session() as session:
            has_child = select(MessageLog.parent_id).where(
                MessageLog.parent_id.isnot(None)  # type: ignore[unresolved-attribute]
            )
            incoming_ids = select(MessageLog.id).where(
                MessageLog.direction == PennyConstants.MessageDirection.INCOMING
            )
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                        MessageLog.id.notin_(has_child),  # type: ignore[unresolved-attribute]
                        MessageLog.parent_id.in_(incoming_ids),  # type: ignore[unresolved-attribute]
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_user_messages(self, sender: str, limit: int = 100) -> list[MessageLog]:
        """Get incoming messages from a specific user, oldest first."""
        with self._session() as session:
            messages = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )
            messages.reverse()
            return messages

    def get_conversation(self, sender: str, limit: int = 20) -> list[MessageLog]:
        """Get recent conversation messages (both directions), oldest first."""
        with self._session() as session:
            incoming = self._get_recent_incoming(session, sender, limit)
            threaded = self._get_threaded_replies(session, incoming)
            autonomous = self._get_autonomous_outgoing(session, sender, limit)
            all_messages = incoming + threaded + autonomous
            all_messages.sort(key=lambda m: m.timestamp)
            return all_messages[-limit:]

    def _get_recent_incoming(self, session: Any, sender: str, limit: int) -> list[MessageLog]:
        """Fetch the most recent incoming messages from a user."""
        return list(
            session.exec(
                select(MessageLog)
                .where(
                    MessageLog.sender == sender,
                    MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    MessageLog.is_reaction == False,  # noqa: E712
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(limit)
            ).all()
        )

    def _get_threaded_replies(self, session: Any, incoming: list[MessageLog]) -> list[MessageLog]:
        """Fetch outgoing messages that are direct replies to the given incoming messages."""
        incoming_ids = [m.id for m in incoming if m.id is not None]
        if not incoming_ids:
            return []
        return list(
            session.exec(
                select(MessageLog).where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.parent_id.in_(incoming_ids),  # type: ignore[unresolved-attribute]
                )
            ).all()
        )

    def _get_autonomous_outgoing(
        self, session: Any, recipient: str, limit: int
    ) -> list[MessageLog]:
        """Fetch autonomous outgoing messages (no parent thread) sent to a user."""
        return list(
            session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.parent_id == None,  # noqa: E711
                    MessageLog.recipient == recipient,
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(limit)
            ).all()
        )

    def get_messages_in_range(
        self, sender: str, start: datetime, end: datetime
    ) -> list[MessageLog]:
        """Get conversation messages within a date range, oldest first.

        Includes both incoming (from sender) and outgoing (to sender) messages.
        """
        with self._session() as session:
            incoming = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == False,  # noqa: E712
                        MessageLog.timestamp >= start,
                        MessageLog.timestamp < end,
                    )
                    .order_by(MessageLog.timestamp)  # type: ignore[unresolved-attribute]
                ).all()
            )
            outgoing = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                        MessageLog.recipient == sender,
                        MessageLog.timestamp >= start,
                        MessageLog.timestamp < end,
                    )
                    .order_by(MessageLog.timestamp)  # type: ignore[unresolved-attribute]
                ).all()
            )
            all_messages = incoming + outgoing
            all_messages.sort(key=lambda m: m.timestamp)
            return all_messages

    def get_reactions_in_range(
        self, sender: str, start: datetime, end: datetime
    ) -> list[MessageLog]:
        """Get reaction messages from a user within a date range, oldest first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == True,  # noqa: E712
                        MessageLog.timestamp >= start,
                        MessageLog.timestamp < end,
                    )
                    .order_by(MessageLog.timestamp)  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_messages_since(
        self, sender: str, since: datetime, limit: int = 100
    ) -> list[MessageLog]:
        """Get conversation messages since a timestamp, oldest first, capped at limit."""
        with self._session() as session:
            incoming = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == False,  # noqa: E712
                        MessageLog.timestamp >= since,
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )
            threaded = self._get_threaded_replies(session, incoming)
            autonomous = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                        MessageLog.parent_id == None,  # noqa: E711
                        MessageLog.recipient == sender,
                        MessageLog.timestamp >= since,
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )
            all_messages = incoming + threaded + autonomous
            all_messages.sort(key=lambda m: m.timestamp)
            return all_messages[-limit:]

    def get_unprocessed(self, sender: str, limit: int) -> list[MessageLog]:
        """Get recent unprocessed non-reaction messages from a specific user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == False,  # noqa: E712
                        MessageLog.processed == False,  # noqa: E712
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def get_user_reactions(self, sender: str, limit: int) -> list[MessageLog]:
        """Get recent unprocessed reactions from a specific user."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                        MessageLog.is_reaction == True,  # noqa: E712
                        MessageLog.processed == False,  # noqa: E712
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def mark_processed(self, message_ids: list[int]) -> None:
        """Mark multiple messages as processed."""
        if not message_ids:
            return
        try:
            with self._session() as session:
                for message_id in message_ids:
                    msg = session.get(MessageLog, message_id)
                    if msg:
                        msg.processed = True
                        session.add(msg)
                session.commit()
                logger.debug("Marked %d messages as processed", len(message_ids))
        except Exception as e:
            logger.error("Failed to mark messages as processed: %s", e)

    def mark_reaction_processed(self, message_id: int) -> None:
        """Mark a reaction message as processed."""
        try:
            with self._session() as session:
                msg = session.get(MessageLog, message_id)
                if msg and msg.is_reaction:
                    msg.processed = True
                    session.add(msg)
                    session.commit()
        except Exception as e:
            logger.error("Failed to mark reaction as processed: %s", e)

    # --- Aggregate queries ---

    def count(self) -> int:
        """Count total number of messages."""
        with self._session() as session:
            from sqlalchemy import func

            return session.exec(select(func.count()).select_from(MessageLog)).one()

    def count_active_threads(self) -> int:
        """Count leaf messages (those with no children)."""
        with self._session() as session:
            from sqlalchemy import func

            has_child = select(MessageLog.parent_id).where(
                MessageLog.parent_id.isnot(None)  # type: ignore[unresolved-attribute]
            )
            return session.exec(
                select(func.count()).select_from(MessageLog).where(MessageLog.id.notin_(has_child))  # type: ignore[unresolved-attribute]
            ).one()

    def get_latest_incoming_time(self, sender: str) -> datetime | None:
        """Get timestamp of the most recent incoming message from a user."""
        with self._session() as session:
            return session.exec(
                select(MessageLog.timestamp)
                .where(
                    MessageLog.sender == sender,
                    MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    MessageLog.is_reaction == False,  # noqa: E712
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def get_latest_autonomous_outgoing_time(self, recipient: str) -> datetime | None:
        """Get timestamp of the most recent autonomous outgoing message to a user."""
        with self._session() as session:
            return session.exec(
                select(MessageLog.timestamp)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.parent_id == None,  # noqa: E711
                    MessageLog.recipient == recipient,
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def count_autonomous_since_last_incoming(self, user: str, after: datetime | None = None) -> int:
        """Count autonomous outgoing messages since the user's last incoming message.

        Args:
            user: The user identifier.
            after: Optional floor timestamp — messages before this are excluded.
                   Used to reset backoff count on service restart.
        """
        latest_incoming = self.get_latest_incoming_time(user)
        # Use the later of last incoming and the floor timestamp
        cutoff = latest_incoming
        if after is not None and (cutoff is None or after > cutoff):
            cutoff = after
        with self._session() as session:
            query = select(func.count(MessageLog.id)).where(
                MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                MessageLog.parent_id == None,  # noqa: E711
                MessageLog.recipient == user,
            )
            if cutoff is not None:
                query = query.where(MessageLog.timestamp > cutoff)
            return session.exec(query).one()

    def get_last_checkin_time(self, prompt_text: str, hours: int = 48) -> datetime | None:
        """Get timestamp of the most recent prompt log containing the check-in prompt."""
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)
        with self._session() as session:
            return session.exec(
                select(PromptLog.timestamp)
                .where(
                    PromptLog.messages.contains(prompt_text),  # type: ignore[unresolved-attribute]
                    PromptLog.timestamp >= cutoff,
                )
                .order_by(PromptLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()

    def get_recent_outgoing_content(
        self, recipient: str, hours: int = 24, limit: int = 20
    ) -> list[str]:
        """Get content of recent outgoing messages for novelty scoring."""
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)
        with self._session() as session:
            messages = session.exec(
                select(MessageLog.content)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.recipient == recipient,
                    MessageLog.timestamp >= cutoff,
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(limit)
            ).all()
            return [m for m in messages if m]

    def get_latest_message_time_in_range(
        self, sender: str, start: datetime, end: datetime
    ) -> datetime | None:
        """Get timestamp of the most recent message (incoming or outgoing) in a range."""
        with self._session() as session:
            incoming_ts = session.exec(
                select(MessageLog.timestamp)
                .where(
                    MessageLog.sender == sender,
                    MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    MessageLog.is_reaction == False,  # noqa: E712
                    MessageLog.timestamp >= start,
                    MessageLog.timestamp < end,
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()
            outgoing_ts = session.exec(
                select(MessageLog.timestamp)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    MessageLog.recipient == sender,
                    MessageLog.timestamp >= start,
                    MessageLog.timestamp < end,
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()
            if incoming_ts and outgoing_ts:
                return max(incoming_ts, outgoing_ts)
            return incoming_ts or outgoing_ts

    def get_first_message_time(self, sender: str) -> datetime | None:
        """Get timestamp of the earliest incoming message from a user."""
        with self._session() as session:
            return session.exec(
                select(MessageLog.timestamp)
                .where(
                    MessageLog.sender == sender,
                    MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    MessageLog.is_reaction == False,  # noqa: E712
                )
                .order_by(MessageLog.timestamp)  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()
