"""Database connection and session management."""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from penny.agent.models import MessageRole
from penny.constants import MessageDirection
from penny.database.models import MessageLog, PromptLog, SearchLog, UserProfile

logger = logging.getLogger(__name__)


class Database:
    """Database manager for Penny's memory."""

    @staticmethod
    def _strip_formatting(text: str) -> str:
        """Strip markdown formatting for quote lookup.

        Signal converts **bold**/etc. to native formatting, so quotes come back
        as plain text. We strip these markers to enable reliable matching.
        """
        # Remove bold/italic markers
        text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
        # Remove strikethrough (both ~~ and ~)
        text = re.sub(r"~{1,2}(.+?)~{1,2}", r"\1", text)
        # Remove monospace
        text = re.sub(r"`(.+?)`", r"\1", text)
        # Normalize tilde operator (U+223C) to regular tilde for consistent matching
        # (Signal channel uses tilde operator to prevent accidental strikethrough)
        text = text.replace("\u223c", "~")
        return text

    def __init__(self, db_path: str):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(f"sqlite:///{db_path}")

        logger.info("Database initialized: %s", db_path)

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Session:
        """Get a database session."""
        return Session(self.engine)

    def log_prompt(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        tools: list[dict] | None = None,
        thinking: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """
        Log a prompt/response exchange with Ollama.

        Args:
            model: Model name used
            messages: Messages sent to the model
            response: Response dict from the model
            tools: Optional tool definitions sent
            thinking: Optional model thinking/reasoning trace
            duration_ms: Optional call duration in milliseconds
        """
        try:
            with self.get_session() as session:
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

    def log_search(
        self,
        query: str,
        response: str,
        duration_ms: int | None = None,
    ) -> None:
        """
        Log a Perplexity search call.

        Args:
            query: The search query
            response: The search response text
            duration_ms: Optional call duration in milliseconds
        """
        try:
            with self.get_session() as session:
                log = SearchLog(
                    query=query,
                    response=response,
                    duration_ms=duration_ms,
                )
                session.add(log)
                session.commit()
                logger.debug("Logged search query: %s", query[:50])
        except Exception as e:
            logger.error("Failed to log search: %s", e)

    def log_message(
        self,
        direction: str,
        sender: str,
        content: str,
        parent_id: int | None = None,
        signal_timestamp: int | None = None,
    ) -> int | None:
        """
        Log a user message or agent response.

        Args:
            direction: "incoming" for user messages, "outgoing" for agent responses
            sender: Who sent the message (phone number or "agent")
            content: The message text
            parent_id: Optional id of the parent message in the thread
            signal_timestamp: Optional Signal message timestamp (ms since epoch)

        Returns:
            The id of the created message, or None on failure
        """
        # Strip formatting from outgoing messages for reliable quote matching
        if direction == MessageDirection.OUTGOING:
            content = self._strip_formatting(content)

        try:
            with self.get_session() as session:
                log = MessageLog(
                    direction=direction,
                    sender=sender,
                    content=content,
                    parent_id=parent_id,
                    signal_timestamp=signal_timestamp,
                )
                session.add(log)
                session.commit()
                session.refresh(log)
                logger.debug("Logged %s message from %s (id=%d)", direction, sender, log.id)
                return log.id
        except Exception as e:
            logger.error("Failed to log message: %s", e)
            return None

    def set_signal_timestamp(self, message_id: int, signal_timestamp: int) -> None:
        """
        Update the Signal timestamp on a message after sending.

        Args:
            message_id: The message ID to update
            signal_timestamp: The Signal timestamp (ms since epoch)
        """
        try:
            with self.get_session() as session:
                msg = session.get(MessageLog, message_id)
                if msg:
                    msg.signal_timestamp = signal_timestamp
                    session.add(msg)
                    session.commit()
                    logger.debug("Set signal_timestamp on message %d", message_id)
        except Exception as e:
            logger.error("Failed to set signal_timestamp: %s", e)

    def get_unsummarized_messages(self) -> list[MessageLog]:
        """Get all outgoing messages that have a parent but no summary yet."""
        with self.get_session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.direction == MessageDirection.OUTGOING,
                        MessageLog.parent_id.isnot(None),  # type: ignore[unresolved-attribute]
                        MessageLog.parent_summary.is_(None),  # type: ignore[unresolved-attribute]
                    )
                    .order_by(MessageLog.timestamp.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def set_parent_summary(self, message_id: int, summary: str) -> None:
        """Store a thread summary on a message."""
        try:
            with self.get_session() as session:
                msg = session.get(MessageLog, message_id)
                if msg:
                    msg.parent_summary = summary
                    session.add(msg)
                    session.commit()
                    logger.debug("Set parent_summary on message %d", message_id)
        except Exception as e:
            logger.error("Failed to set parent_summary: %s", e)

    def get_conversation_leaves(self) -> list[MessageLog]:
        """Get outgoing leaf messages eligible for spontaneous continuation.

        Returns outgoing messages that have no children and whose parent is an
        incoming (user) message. An already-continued thread has an outgoing
        parent instead, so it's naturally excluded.
        """
        with self.get_session() as session:
            # Subquery: all IDs that have a child message
            has_child = select(MessageLog.parent_id).where(
                MessageLog.parent_id.isnot(None)  # type: ignore[unresolved-attribute]
            )
            # Subquery: IDs of incoming messages
            incoming_ids = select(MessageLog.id).where(
                MessageLog.direction == MessageDirection.INCOMING
            )
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.direction == MessageDirection.OUTGOING,
                        MessageLog.id.notin_(has_child),  # type: ignore[unresolved-attribute]
                        MessageLog.parent_id.in_(incoming_ids),  # type: ignore[unresolved-attribute]
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def find_outgoing_by_content(self, content: str) -> MessageLog | None:
        """
        Find the most recent outgoing message matching the given content.
        Used to look up which agent response a user is quoting.

        Signal truncates quoted text, so we use prefix matching (startswith).

        Args:
            content: The quoted text to search for

        Returns:
            The matching MessageLog, or None
        """
        content = self._strip_formatting(content)
        with self.get_session() as session:
            return session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == MessageDirection.OUTGOING,
                    MessageLog.content.startswith(content),
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
            ).first()

    def get_thread_context(
        self, quoted_text: str
    ) -> tuple[int | None, list[tuple[str, str]] | None]:
        """
        Look up a quoted message and return its id and conversation context.
        Uses the cached summary if available, otherwise walks the parent chain.

        Args:
            quoted_text: The text the user quoted/replied to

        Returns:
            Tuple of (parent_id, history) where history is a list of (role, content) tuples,
            or (None, None) if the quoted message wasn't found.
        """
        parent_msg = self.find_outgoing_by_content(quoted_text)
        if not parent_msg:
            logger.warning(
                "Could not find quoted message in database, thread context will be empty"
            )
            return None, None

        if parent_msg.parent_summary:
            history = [
                (MessageRole.SYSTEM, f"Previous conversation summary: {parent_msg.parent_summary}")
            ]
            logger.info("Using cached thread summary for context")
        else:
            assert parent_msg.id is not None
            thread = self._walk_thread(parent_msg.id)
            history = [
                (
                    MessageRole.USER
                    if m.direction == MessageDirection.INCOMING
                    else MessageRole.ASSISTANT,
                    m.content,
                )
                for m in thread
            ]
            logger.info("Built thread history with %d messages", len(history))

        return parent_msg.id, history

    def _walk_thread(self, message_id: int, limit: int = 20) -> list[MessageLog]:
        """
        Walk up the parent chain from a message.
        Returns messages in chronological order (oldest first).
        """
        history = []
        with self.get_session() as session:
            current_id = message_id
            while current_id is not None and len(history) < limit:
                msg = session.get(MessageLog, current_id)
                if msg is None:
                    break
                history.append(msg)
                current_id = msg.parent_id

        history.reverse()
        return history

    def get_user_messages(self, sender: str, limit: int = 100) -> list[MessageLog]:
        """
        Get incoming messages from a specific user.

        Args:
            sender: The user's identifier (phone number, discord ID, etc.)
            limit: Maximum number of messages to return

        Returns:
            Most recent messages ordered by timestamp ascending (oldest first)
        """
        with self.get_session() as session:
            # Get newest messages first, then reverse for chronological order
            messages = list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == MessageDirection.INCOMING,
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )
            messages.reverse()  # Return in chronological order (oldest first)
            return messages

    def get_users_needing_profile_update(self) -> list[str]:
        """
        Get senders who have messages newer than their last profile update.

        Returns:
            List of sender IDs that need profile updates
        """
        with self.get_session() as session:
            # Get all distinct incoming message senders
            all_senders = list(
                session.exec(
                    select(MessageLog.sender)
                    .where(MessageLog.direction == MessageDirection.INCOMING)
                    .distinct()
                ).all()
            )

            users_needing_update = []
            for sender in all_senders:
                profile = session.exec(
                    select(UserProfile).where(UserProfile.sender == sender)
                ).first()

                latest_msg = session.exec(
                    select(MessageLog.timestamp)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == MessageDirection.INCOMING,
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(1)
                ).first()

                if latest_msg and (not profile or profile.last_message_timestamp < latest_msg):
                    users_needing_update.append(sender)

            return users_needing_update

    def get_user_profile(self, sender: str) -> UserProfile | None:
        """
        Get the cached profile for a user.

        Args:
            sender: The user's identifier

        Returns:
            The UserProfile if it exists, None otherwise
        """
        with self.get_session() as session:
            return session.exec(select(UserProfile).where(UserProfile.sender == sender)).first()

    def get_users_with_profiles(self) -> list[str]:
        """
        Get all users who have a profile.

        Returns:
            List of sender IDs that have profiles
        """
        with self.get_session() as session:
            profiles = session.exec(select(UserProfile.sender)).all()
            return list(profiles)

    def save_user_profile(
        self,
        sender: str,
        profile_text: str,
        last_message_timestamp: "datetime",
    ) -> None:
        """
        Create or update a user profile.

        Args:
            sender: The user's identifier
            profile_text: The generated profile content
            last_message_timestamp: Timestamp of the newest message included
        """
        try:
            with self.get_session() as session:
                existing = session.exec(
                    select(UserProfile).where(UserProfile.sender == sender)
                ).first()

                if existing:
                    existing.profile_text = profile_text
                    existing.updated_at = datetime.now(UTC)
                    existing.last_message_timestamp = last_message_timestamp
                    session.add(existing)
                else:
                    profile = UserProfile(
                        sender=sender,
                        profile_text=profile_text,
                        last_message_timestamp=last_message_timestamp,
                    )
                    session.add(profile)

                session.commit()
                logger.debug("Saved profile for %s", sender)
        except Exception as e:
            logger.error("Failed to save profile: %s", e)
