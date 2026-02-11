"""Database connection and session management."""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from penny.agents.models import MessageRole
from penny.constants import MessageDirection
from penny.database.models import (
    CommandLog,
    MessageLog,
    Preference,
    PromptLog,
    SearchLog,
    UserInfo,
)

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
        external_id: str | None = None,
        is_reaction: bool = False,
    ) -> int | None:
        """
        Log a user message or agent response.

        Args:
            direction: "incoming" for user messages, "outgoing" for agent responses
            sender: Who sent the message (phone number or "agent")
            content: The message text
            parent_id: Optional id of the parent message in the thread
            signal_timestamp: Optional Signal message timestamp (ms since epoch)
            external_id: Optional platform-specific message ID for reaction lookup
            is_reaction: True if this is a reaction message

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

    def set_external_id(self, message_id: int, external_id: str) -> None:
        """
        Update the external ID on a message after sending.

        Args:
            message_id: The message ID to update
            external_id: The platform-specific message ID
        """
        try:
            with self.get_session() as session:
                msg = session.get(MessageLog, message_id)
                if msg:
                    msg.external_id = external_id
                    session.add(msg)
                    session.commit()
                    logger.debug("Set external_id on message %d", message_id)
        except Exception as e:
            logger.error("Failed to set external_id: %s", e)

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

    def get_message_by_id(self, message_id: int) -> MessageLog | None:
        """
        Get a message by its database ID.

        Args:
            message_id: The primary key ID of the message

        Returns:
            The matching MessageLog, or None
        """
        with self.get_session() as session:
            return session.get(MessageLog, message_id)

    def find_message_by_external_id(self, external_id: str) -> MessageLog | None:
        """
        Find a message by its platform-specific external ID.
        Used to look up which message was reacted to.

        Args:
            external_id: The platform-specific message ID (Signal timestamp or Discord message ID)

        Returns:
            The matching MessageLog, or None
        """
        with self.get_session() as session:
            return session.exec(
                select(MessageLog).where(MessageLog.external_id == external_id)
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

    def get_unprocessed_messages(self, sender: str, limit: int) -> list[MessageLog]:
        """
        Get recent unprocessed non-reaction messages from a specific user.

        Args:
            sender: The user's identifier (phone number, discord ID, etc.)
            limit: Maximum number of messages to return

        Returns:
            Unprocessed messages ordered by timestamp ascending (oldest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == MessageDirection.INCOMING,
                        MessageLog.is_reaction == False,  # noqa: E712
                        MessageLog.processed == False,  # noqa: E712
                    )
                    .order_by(MessageLog.timestamp.asc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def get_user_reactions(self, sender: str, limit: int) -> list[MessageLog]:
        """
        Get recent unprocessed reactions from a specific user.

        Args:
            sender: The user's identifier (phone number, discord ID, etc.)
            limit: Maximum number of reactions to return

        Returns:
            Most recent unprocessed reactions ordered by timestamp descending (newest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(MessageLog)
                    .where(
                        MessageLog.sender == sender,
                        MessageLog.direction == MessageDirection.INCOMING,
                        MessageLog.is_reaction == True,  # noqa: E712
                        MessageLog.processed == False,  # noqa: E712
                    )
                    .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def mark_messages_processed(self, message_ids: list[int]) -> None:
        """
        Mark multiple messages as processed.

        Args:
            message_ids: List of message IDs to mark as processed
        """
        if not message_ids:
            return
        try:
            with self.get_session() as session:
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
        """
        Mark a reaction message as processed.

        Args:
            message_id: The ID of the reaction message to mark
        """
        try:
            with self.get_session() as session:
                msg = session.get(MessageLog, message_id)
                if msg and msg.is_reaction:
                    msg.processed = True
                    session.add(msg)
                    session.commit()
                    logger.debug("Marked reaction %d as processed", message_id)
        except Exception as e:
            logger.error("Failed to mark reaction as processed: %s", e)

    def get_all_senders(self) -> list[str]:
        """
        Get all unique senders who have sent messages.

        Returns:
            List of unique sender IDs from incoming messages
        """
        with self.get_session() as session:
            senders = session.exec(
                select(MessageLog.sender)
                .where(MessageLog.direction == MessageDirection.INCOMING)
                .distinct()
            ).all()
            return list(senders)

    def get_user_info(self, sender: str) -> UserInfo | None:
        """
        Get the basic user info for a user.

        Args:
            sender: The user's identifier

        Returns:
            The UserInfo if it exists, None otherwise
        """
        with self.get_session() as session:
            return session.exec(select(UserInfo).where(UserInfo.sender == sender)).first()

    def save_user_info(
        self,
        sender: str,
        name: str,
        location: str,
        timezone: str,
        date_of_birth: str,
    ) -> None:
        """
        Create or update user info.

        Args:
            sender: The user's identifier
            name: User's preferred name
            location: Natural language location
            timezone: IANA timezone string
            date_of_birth: Date of birth in YYYY-MM-DD format
        """
        try:
            with self.get_session() as session:
                existing = session.exec(select(UserInfo).where(UserInfo.sender == sender)).first()

                if existing:
                    existing.name = name
                    existing.location = location
                    existing.timezone = timezone
                    existing.date_of_birth = date_of_birth
                    existing.updated_at = datetime.now(UTC)
                    session.add(existing)
                else:
                    info = UserInfo(
                        sender=sender,
                        name=name,
                        location=location,
                        timezone=timezone,
                        date_of_birth=date_of_birth,
                    )
                    session.add(info)

                session.commit()
                logger.debug("Saved user info for %s", sender)
        except Exception as e:
            logger.error("Failed to save user info: %s", e)

    def count_messages(self) -> int:
        """
        Count total number of messages (incoming + outgoing).

        Returns:
            Total message count
        """
        with self.get_session() as session:
            from sqlalchemy import func

            return session.exec(select(func.count()).select_from(MessageLog)).one()

    def count_active_threads(self) -> int:
        """
        Count number of active conversation threads.

        A thread is active if it's a leaf message (has no children).

        Returns:
            Number of active threads
        """
        with self.get_session() as session:
            from sqlalchemy import func

            # Get all message IDs that appear as parent_id (have children)
            has_child = select(MessageLog.parent_id).where(
                MessageLog.parent_id.isnot(None)  # type: ignore[unresolved-attribute]
            )
            # Count messages that are NOT in that set (leaves)
            return session.exec(
                select(func.count())
                .select_from(MessageLog)
                .where(
                    MessageLog.id.notin_(has_child)  # type: ignore[unresolved-attribute]
                )
            ).one()

    def log_command(
        self,
        user: str,
        channel_type: str,
        command_name: str,
        command_args: str,
        response: str,
        error: str | None = None,
    ) -> None:
        """
        Log a command invocation.

        Args:
            user: User who invoked the command
            channel_type: Channel type ("signal" or "discord")
            command_name: Name of the command (without / prefix)
            command_args: Arguments passed to the command
            response: Response text sent to user
            error: Error message if command failed
        """
        try:
            with self.get_session() as session:
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

    def get_preferences(self, user: str, pref_type: str) -> list[Preference]:
        """
        Get all preferences of a specific type for a user.

        Args:
            user: User identifier (phone number or Discord user ID)
            pref_type: Preference type ("like" or "dislike")

        Returns:
            List of matching Preference objects
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Preference)
                    .where(
                        Preference.user == user,
                        Preference.type == pref_type,
                    )
                    .order_by(Preference.created_at.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def add_preference(self, user: str, topic: str, pref_type: str) -> bool:
        """
        Add a preference for a user.

        Args:
            user: User identifier (phone number or Discord user ID)
            topic: The topic/phrase to add
            pref_type: Preference type ("like" or "dislike")

        Returns:
            True if added, False if already exists
        """
        try:
            with self.get_session() as session:
                # Check if already exists
                existing = session.exec(
                    select(Preference).where(
                        Preference.user == user,
                        Preference.topic == topic,
                        Preference.type == pref_type,
                    )
                ).first()

                if existing:
                    return False

                pref = Preference(
                    user=user,
                    topic=topic,
                    type=pref_type,
                )
                session.add(pref)
                session.commit()
                logger.debug("Added %s preference for %s: %s", pref_type, user, topic)
                return True
        except Exception as e:
            logger.error("Failed to add preference: %s", e)
            return False

    def remove_preference(self, user: str, topic: str, pref_type: str) -> bool:
        """
        Remove a preference for a user.

        Args:
            user: User identifier (phone number or Discord user ID)
            topic: The topic/phrase to remove
            pref_type: Preference type ("like" or "dislike")

        Returns:
            True if removed, False if not found
        """
        try:
            with self.get_session() as session:
                pref = session.exec(
                    select(Preference).where(
                        Preference.user == user,
                        Preference.topic == topic,
                        Preference.type == pref_type,
                    )
                ).first()

                if not pref:
                    return False

                session.delete(pref)
                session.commit()
                logger.debug("Removed %s preference for %s: %s", pref_type, user, topic)
                return True
        except Exception as e:
            logger.error("Failed to remove preference: %s", e)
            return False

    def find_conflicting_preference(
        self, user: str, topic: str, pref_type: str
    ) -> Preference | None:
        """
        Find a conflicting preference (opposite type with same topic).

        Args:
            user: User identifier (phone number or Discord user ID)
            topic: The topic/phrase to check
            pref_type: Preference type to check against

        Returns:
            The conflicting Preference if found, None otherwise
        """
        opposite_type = "dislike" if pref_type == "like" else "like"
        with self.get_session() as session:
            return session.exec(
                select(Preference).where(
                    Preference.user == user,
                    Preference.topic == topic,
                    Preference.type == opposite_type,
                )
            ).first()

    def move_preference(self, user: str, topic: str, from_type: str, to_type: str) -> bool:
        """
        Move a preference from one type to another (e.g., like → dislike).

        Args:
            user: User identifier (phone number or Discord user ID)
            topic: The topic/phrase to move
            from_type: Source preference type
            to_type: Target preference type

        Returns:
            True if moved successfully, False otherwise
        """
        try:
            with self.get_session() as session:
                # Find the existing preference
                existing = session.exec(
                    select(Preference).where(
                        Preference.user == user,
                        Preference.topic == topic,
                        Preference.type == from_type,
                    )
                ).first()

                if not existing:
                    return False

                # Remove it
                session.delete(existing)
                session.flush()

                # Add new one with opposite type
                new_pref = Preference(
                    user=user,
                    topic=topic,
                    type=to_type,
                )
                session.add(new_pref)
                session.commit()
                logger.debug(
                    "Moved preference for %s: %s from %s to %s", user, topic, from_type, to_type
                )
                return True
        except Exception as e:
            logger.error("Failed to move preference: %s", e)
            return False
