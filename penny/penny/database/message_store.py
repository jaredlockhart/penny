"""Message store — logging, threading, and queries for messages."""

import logging
import re
from datetime import datetime

from sqlmodel import Session, select

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

    def get_latest_interaction_time(self, sender: str) -> datetime | None:
        """Get timestamp of most recent user interaction (for backoff logic)."""
        return self.get_latest_incoming_time(sender)
