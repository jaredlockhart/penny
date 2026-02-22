"""Database connection and session management."""

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from penny.agents.models import MessageRole
from penny.constants import PennyConstants
from penny.database.models import (
    CommandLog,
    Engagement,
    Entity,
    Fact,
    LearnPrompt,
    MessageLog,
    MuteState,
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
        trigger: str = "user_message",
        learn_prompt_id: int | None = None,
    ) -> None:
        """
        Log a Perplexity search call.

        Args:
            query: The search query
            response: The search response text
            duration_ms: Optional call duration in milliseconds
            trigger: What triggered this search (SearchTrigger enum value)
            learn_prompt_id: Optional LearnPrompt FK for /learn command searches
        """
        try:
            with self.get_session() as session:
                log = SearchLog(
                    query=query,
                    response=response,
                    duration_ms=duration_ms,
                    trigger=trigger,
                    learn_prompt_id=learn_prompt_id,
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
        if direction == PennyConstants.MessageDirection.OUTGOING:
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
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
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
        Walks the parent chain to build the full thread history.

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
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
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
            Unprocessed messages ordered by timestamp descending (newest first)
        """
        with self.get_session() as session:
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
                        MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
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
                .where(MessageLog.direction == PennyConstants.MessageDirection.INCOMING)
                .distinct()
            ).all()
            return list(senders)

    def get_latest_incoming_message_time(self, sender: str) -> datetime | None:
        """Get the timestamp of the most recent incoming message from a user.

        Args:
            sender: The user's identifier

        Returns:
            Timestamp of the most recent incoming message, or None if no messages
        """
        with self.get_session() as session:
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

    def get_latest_user_interaction_time(self, sender: str) -> datetime | None:
        """Get the timestamp of the most recent user interaction (message or command).

        Args:
            sender: The user's identifier

        Returns:
            Most recent timestamp from either MessageLog or CommandLog, or None
        """
        msg_time = self.get_latest_incoming_message_time(sender)
        with self.get_session() as session:
            cmd_time = session.exec(
                select(CommandLog.timestamp)
                .where(CommandLog.user == sender)
                .order_by(CommandLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()

        if msg_time is None:
            return cmd_time
        if cmd_time is None:
            return msg_time
        return max(msg_time, cmd_time)

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

    # --- Entity knowledge base methods ---

    def get_entity(self, entity_id: int) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: Entity primary key

        Returns:
            The Entity if found, None otherwise
        """
        with self.get_session() as session:
            return session.get(Entity, entity_id)

    def get_or_create_entity(self, user: str, name: str) -> Entity | None:
        """
        Get an existing entity or create a new one.

        Args:
            user: User identifier
            name: Entity name (will be lowercased)

        Returns:
            The Entity (existing or newly created), or None on failure
        """
        name = name.lower().strip()
        try:
            with self.get_session() as session:
                existing = session.exec(
                    select(Entity).where(
                        Entity.user == user,
                        Entity.name == name,
                    )
                ).first()

                if existing:
                    existing.updated_at = datetime.now(UTC)
                    session.add(existing)
                    session.commit()
                    session.refresh(existing)
                    return existing

                entity = Entity(
                    user=user,
                    name=name,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                session.add(entity)
                session.commit()
                session.refresh(entity)
                logger.debug("Created entity '%s' for user %s", name, user)
                return entity
        except Exception as e:
            logger.error("Failed to get_or_create entity: %s", e)
            return None

    def get_user_entities(self, user: str) -> list[Entity]:
        """
        Get all entities for a user.

        Args:
            user: User identifier

        Returns:
            List of entities ordered by updated_at descending (most recent first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Entity).where(Entity.user == user).order_by(Entity.updated_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_user_entities_with_embeddings(self, user: str) -> list[Entity]:
        """
        Get all entities for a user that have embeddings.

        Args:
            user: User identifier

        Returns:
            List of entities with non-null embeddings
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Entity).where(
                        Entity.user == user,
                        Entity.embedding != None,  # noqa: E711
                    )
                ).all()
            )

    def delete_entity(self, entity_id: int) -> bool:
        """
        Delete an entity and its associated facts and engagements.

        Args:
            entity_id: Entity primary key

        Returns:
            True if entity existed and was deleted, False otherwise
        """
        try:
            with self.get_session() as session:
                entity = session.get(Entity, entity_id)
                if not entity:
                    return False

                # Delete associated engagements first
                engagements = list(
                    session.exec(select(Engagement).where(Engagement.entity_id == entity_id)).all()
                )
                for engagement in engagements:
                    session.delete(engagement)

                # Delete associated facts
                facts = list(session.exec(select(Fact).where(Fact.entity_id == entity_id)).all())
                for fact in facts:
                    session.delete(fact)

                session.delete(entity)
                session.commit()
                logger.debug(
                    "Deleted entity %d (%d facts, %d engagements)",
                    entity_id,
                    len(facts),
                    len(engagements),
                )
                return True
        except Exception as e:
            logger.error("Failed to delete entity: %s", e)
            return False

    # --- Fact methods ---

    def add_fact(
        self,
        entity_id: int,
        content: str,
        source_url: str | None = None,
        source_search_log_id: int | None = None,
        source_message_id: int | None = None,
        embedding: bytes | None = None,
        notified_at: datetime | None = None,
    ) -> Fact | None:
        """
        Add a fact to an entity.

        Args:
            entity_id: Entity primary key
            content: The fact text
            source_url: URL where the fact was found
            source_search_log_id: SearchLog ID that produced this fact
            source_message_id: MessageLog ID that produced this fact
            embedding: Serialized embedding vector (optional)
            notified_at: Pre-mark as notified (e.g. user-sourced facts)

        Returns:
            The created Fact, or None on failure
        """
        try:
            with self.get_session() as session:
                fact = Fact(
                    entity_id=entity_id,
                    content=content,
                    source_url=source_url,
                    source_search_log_id=source_search_log_id,
                    source_message_id=source_message_id,
                    embedding=embedding,
                    notified_at=notified_at,
                )
                session.add(fact)

                # Touch the parent entity's updated_at
                entity = session.get(Entity, entity_id)
                if entity:
                    entity.updated_at = datetime.now(UTC)
                    session.add(entity)

                session.commit()
                session.refresh(fact)
                logger.debug("Added fact to entity %d: %s", entity_id, content[:50])
                return fact
        except Exception as e:
            logger.error("Failed to add fact to entity %d: %s", entity_id, e)
            return None

    def get_entity_facts(self, entity_id: int) -> list[Fact]:
        """
        Get all facts for an entity.

        Args:
            entity_id: Entity primary key

        Returns:
            List of facts ordered by learned_at ascending (oldest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Fact).where(Fact.entity_id == entity_id).order_by(Fact.learned_at.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    # --- Search extraction tracking ---

    def get_unprocessed_search_logs(self, limit: int) -> list[SearchLog]:
        """
        Get SearchLog entries that haven't been processed for entity extraction.

        Grouped by learn prompt (oldest first) so all searches for one
        /learn topic complete before moving to the next. Non-learn searches
        (NULL learn_prompt_id) come last.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of unprocessed SearchLog entries, grouped by learn prompt (oldest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(SearchLog)
                    .where(SearchLog.extracted == False)  # noqa: E712
                    .order_by(
                        SearchLog.learn_prompt_id.desc(),  # type: ignore[unresolved-attribute]
                        SearchLog.timestamp.asc(),  # type: ignore[unresolved-attribute]
                    )
                    .limit(limit)
                ).all()
            )

    def mark_search_extracted(self, search_log_id: int) -> None:
        """
        Mark a SearchLog entry as processed for entity extraction.

        Args:
            search_log_id: SearchLog primary key
        """
        try:
            with self.get_session() as session:
                search_log = session.get(SearchLog, search_log_id)
                if search_log:
                    search_log.extracted = True
                    session.add(search_log)
                    session.commit()
        except Exception as e:
            logger.error("Failed to mark search %d as extracted: %s", search_log_id, e)

    def find_sender_for_timestamp(self, timestamp: datetime) -> str | None:
        """
        Find the sender of the most recent incoming message near a timestamp.
        Used to associate SearchLog entries with a user.

        The incoming message may be logged slightly after the search
        (same request flow), so we allow a small buffer after the timestamp.

        Args:
            timestamp: The timestamp to search around

        Returns:
            Sender ID, or None if no messages found
        """
        buffer = timedelta(minutes=5)
        with self.get_session() as session:
            msg = session.exec(
                select(MessageLog.sender)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.INCOMING,
                    MessageLog.timestamp <= timestamp + buffer,
                )
                .order_by(MessageLog.timestamp.desc())  # type: ignore[unresolved-attribute]
                .limit(1)
            ).first()
            return msg

    # --- Entity merge ---

    # --- LearnPrompt methods ---

    def create_learn_prompt(
        self,
        user: str,
        prompt_text: str,
        searches_remaining: int = 0,
    ) -> LearnPrompt | None:
        """Create a new LearnPrompt.

        Args:
            user: User identifier
            prompt_text: Original user text
            searches_remaining: Number of searches to execute

        Returns:
            The created LearnPrompt, or None on failure
        """
        try:
            with self.get_session() as session:
                learn_prompt = LearnPrompt(
                    user=user,
                    prompt_text=prompt_text,
                    searches_remaining=searches_remaining,
                )
                session.add(learn_prompt)
                session.commit()
                session.refresh(learn_prompt)
                logger.debug("Created LearnPrompt %d for user %s", learn_prompt.id, user)
                return learn_prompt
        except Exception as e:
            logger.error("Failed to create LearnPrompt: %s", e)
            return None

    def get_search_log(self, search_log_id: int) -> SearchLog | None:
        """Get a SearchLog by ID.

        Args:
            search_log_id: SearchLog primary key

        Returns:
            The SearchLog if found, None otherwise
        """
        with self.get_session() as session:
            return session.get(SearchLog, search_log_id)

    def get_learn_prompt(self, learn_prompt_id: int) -> LearnPrompt | None:
        """Get a LearnPrompt by ID.

        Args:
            learn_prompt_id: LearnPrompt primary key

        Returns:
            The LearnPrompt if found, None otherwise
        """
        with self.get_session() as session:
            return session.get(LearnPrompt, learn_prompt_id)

    def get_next_active_learn_prompt(self) -> LearnPrompt | None:
        """Get the next active LearnPrompt to process (oldest first, across all users).

        Returns:
            The oldest active LearnPrompt, or None if no pending work.
        """
        with self.get_session() as session:
            return session.exec(
                select(LearnPrompt)
                .where(LearnPrompt.status == PennyConstants.LearnPromptStatus.ACTIVE)
                .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                .limit(1)
            ).first()

    def update_learn_prompt_status(self, learn_prompt_id: int, status: str) -> None:
        """Update the status of a LearnPrompt.

        Args:
            learn_prompt_id: LearnPrompt primary key
            status: New status (LearnPromptStatus enum value)
        """
        try:
            with self.get_session() as session:
                learn_prompt = session.get(LearnPrompt, learn_prompt_id)
                if learn_prompt:
                    learn_prompt.status = status
                    learn_prompt.updated_at = datetime.now(UTC)
                    session.add(learn_prompt)
                    session.commit()
                    logger.debug("Updated LearnPrompt %d status to %s", learn_prompt_id, status)
        except Exception as e:
            logger.error("Failed to update LearnPrompt %d status: %s", learn_prompt_id, e)

    def get_active_learn_prompts(self, user: str) -> list[LearnPrompt]:
        """Get all active LearnPrompts for a user.

        Args:
            user: User identifier

        Returns:
            List of active LearnPrompts ordered by created_at ascending (oldest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(LearnPrompt)
                    .where(
                        LearnPrompt.user == user,
                        LearnPrompt.status == "active",
                    )
                    .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def get_user_learn_prompts(self, user: str) -> list[LearnPrompt]:
        """Get all LearnPrompts for a user.

        Args:
            user: User identifier

        Returns:
            List of all LearnPrompts ordered by created_at ascending (oldest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(LearnPrompt)
                    .where(LearnPrompt.user == user)
                    .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def get_search_logs_by_learn_prompt(self, learn_prompt_id: int) -> list[SearchLog]:
        """Get all SearchLogs linked to a LearnPrompt.

        Args:
            learn_prompt_id: LearnPrompt primary key

        Returns:
            List of SearchLogs ordered by timestamp ascending
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(SearchLog)
                    .where(SearchLog.learn_prompt_id == learn_prompt_id)
                    .order_by(SearchLog.timestamp.asc())  # type: ignore[union-attr]
                ).all()
            )

    def get_facts_by_search_log_ids(self, search_log_ids: list[int]) -> list[Fact]:
        """Get all Facts linked to any of the given SearchLog IDs.

        Args:
            search_log_ids: List of SearchLog primary keys

        Returns:
            List of Facts with source_search_log_id in the given list
        """
        if not search_log_ids:
            return []
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Fact).where(
                        Fact.source_search_log_id.in_(search_log_ids)  # type: ignore[union-attr]
                    )
                ).all()
            )

    def decrement_learn_prompt_searches(self, learn_prompt_id: int) -> None:
        """Decrement searches_remaining by 1 on a LearnPrompt.

        Args:
            learn_prompt_id: LearnPrompt primary key
        """
        try:
            with self.get_session() as session:
                learn_prompt = session.get(LearnPrompt, learn_prompt_id)
                if learn_prompt and learn_prompt.searches_remaining > 0:
                    learn_prompt.searches_remaining -= 1
                    learn_prompt.updated_at = datetime.now(UTC)
                    session.add(learn_prompt)
                    session.commit()
        except Exception as e:
            logger.error("Failed to decrement LearnPrompt %d searches: %s", learn_prompt_id, e)

    def update_learn_prompt_searches_remaining(
        self, learn_prompt_id: int, searches_remaining: int
    ) -> None:
        """Set the searches_remaining count on a LearnPrompt.

        Args:
            learn_prompt_id: LearnPrompt primary key
            searches_remaining: New count of remaining searches
        """
        try:
            with self.get_session() as session:
                learn_prompt = session.get(LearnPrompt, learn_prompt_id)
                if learn_prompt:
                    learn_prompt.searches_remaining = searches_remaining
                    learn_prompt.updated_at = datetime.now(UTC)
                    session.add(learn_prompt)
                    session.commit()
        except Exception as e:
            logger.error(
                "Failed to update LearnPrompt %d searches_remaining: %s",
                learn_prompt_id,
                e,
            )

    def get_unannounced_completed_learn_prompts(self, user: str) -> list[LearnPrompt]:
        """Get completed LearnPrompts that haven't had a completion announcement sent.

        Args:
            user: User identifier

        Returns:
            List of completed, unannounced LearnPrompts ordered by
            created_at ascending (oldest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(LearnPrompt)
                    .where(
                        LearnPrompt.user == user,
                        LearnPrompt.status == "completed",
                        LearnPrompt.announced_at.is_(None),  # type: ignore[union-attr]
                    )
                    .order_by(LearnPrompt.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def mark_learn_prompt_announced(self, learn_prompt_id: int) -> None:
        """Mark a LearnPrompt as announced by setting announced_at.

        Args:
            learn_prompt_id: LearnPrompt primary key
        """
        try:
            with self.get_session() as session:
                learn_prompt = session.get(LearnPrompt, learn_prompt_id)
                if learn_prompt:
                    learn_prompt.announced_at = datetime.now(UTC)
                    session.add(learn_prompt)
                    session.commit()
        except Exception as e:
            logger.error(
                "Failed to mark LearnPrompt %d as announced: %s",
                learn_prompt_id,
                e,
            )

    def delete_learn_prompt(self, learn_prompt_id: int) -> list[tuple[str, int]]:
        """Delete a LearnPrompt and all entities/facts acquired from its searches.

        Deletes in order: engagements → facts → entities → search logs → learn prompt.
        An entity is deleted if ANY of its facts came from this learn prompt's searches.

        Args:
            learn_prompt_id: LearnPrompt primary key

        Returns:
            List of (entity_name, fact_count) for each deleted entity
        """
        try:
            with self.get_session() as session:
                learn_prompt = session.get(LearnPrompt, learn_prompt_id)
                if not learn_prompt:
                    return []

                # Find all search logs for this learn prompt
                search_logs = list(
                    session.exec(
                        select(SearchLog).where(SearchLog.learn_prompt_id == learn_prompt_id)
                    ).all()
                )
                search_log_ids = [sl.id for sl in search_logs if sl.id is not None]

                # Find all facts from those searches
                facts: list[Fact] = []
                if search_log_ids:
                    facts = list(
                        session.exec(
                            select(Fact).where(
                                Fact.source_search_log_id.in_(search_log_ids)  # type: ignore[union-attr]
                            )
                        ).all()
                    )

                # Collect unique entity IDs to delete
                entity_ids = {f.entity_id for f in facts}

                # Build per-entity summary before deleting
                deleted_entities: list[tuple[str, int]] = []
                for entity_id in entity_ids:
                    entity = session.get(Entity, entity_id)
                    if not entity:
                        continue

                    # Delete engagements
                    engagements = list(
                        session.exec(
                            select(Engagement).where(Engagement.entity_id == entity_id)
                        ).all()
                    )
                    for eng in engagements:
                        session.delete(eng)

                    # Delete ALL facts for this entity
                    entity_facts = list(
                        session.exec(select(Fact).where(Fact.entity_id == entity_id)).all()
                    )
                    for fact in entity_facts:
                        session.delete(fact)

                    deleted_entities.append((entity.name, len(entity_facts)))
                    session.delete(entity)

                # Delete search logs
                for sl in search_logs:
                    session.delete(sl)

                # Delete the learn prompt
                session.delete(learn_prompt)
                session.commit()

                logger.debug(
                    "Deleted LearnPrompt %d (%d entities)",
                    learn_prompt_id,
                    len(deleted_entities),
                )
                return deleted_entities
        except Exception as e:
            logger.error("Failed to delete LearnPrompt %d: %s", learn_prompt_id, e)
            return []

    # --- Embedding methods ---

    def update_entity_embedding(self, entity_id: int, embedding: bytes) -> None:
        """Update the embedding for an entity.

        Args:
            entity_id: Entity primary key
            embedding: Serialized embedding vector
        """
        try:
            with self.get_session() as session:
                entity = session.get(Entity, entity_id)
                if entity:
                    entity.embedding = embedding
                    session.add(entity)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update entity %d embedding: %s", entity_id, e)

    def update_entity_tagline(self, entity_id: int, tagline: str) -> None:
        """Update the tagline for an entity.

        Args:
            entity_id: Entity primary key
            tagline: Short disambiguating summary
        """
        try:
            with self.get_session() as session:
                entity = session.get(Entity, entity_id)
                if entity:
                    entity.tagline = tagline
                    session.add(entity)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update entity %d tagline: %s", entity_id, e)

    def update_fact_embedding(self, fact_id: int, embedding: bytes) -> None:
        """Update the embedding for a fact.

        Args:
            fact_id: Fact primary key
            embedding: Serialized embedding vector
        """
        try:
            with self.get_session() as session:
                fact = session.get(Fact, fact_id)
                if fact:
                    fact.embedding = embedding
                    session.add(fact)
                    session.commit()
        except Exception as e:
            logger.error("Failed to update fact %d embedding: %s", fact_id, e)

    def get_entities_without_embeddings(self, limit: int) -> list[Entity]:
        """Get entities that don't have embeddings yet.

        Args:
            limit: Maximum number of entities to return

        Returns:
            List of Entity objects without embeddings
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Entity)
                    .where(Entity.embedding == None)  # noqa: E711
                    .order_by(Entity.created_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def get_facts_without_embeddings(self, limit: int) -> list[Fact]:
        """Get facts that don't have embeddings yet.

        Args:
            limit: Maximum number of facts to return

        Returns:
            List of Fact objects without embeddings
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Fact)
                    .where(Fact.embedding == None)  # noqa: E711
                    .order_by(Fact.learned_at.desc())  # type: ignore[unresolved-attribute]
                    .limit(limit)
                ).all()
            )

    def get_unnotified_facts(self, user: str) -> list[Fact]:
        """Get facts that haven't been communicated to a user yet.

        Returns facts where notified_at IS NULL for entities owned by this user,
        ordered by learned_at descending (newest first).
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Fact)
                    .join(Entity, Fact.entity_id == Entity.id)  # type: ignore[invalid-argument-type]
                    .where(Entity.user == user, Fact.notified_at == None)  # noqa: E711
                    .order_by(Fact.learned_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def mark_facts_notified(self, fact_ids: list[int]) -> None:
        """Mark facts as notified (communicated to user).

        Args:
            fact_ids: List of fact IDs to mark
        """
        if not fact_ids:
            return
        try:
            now = datetime.now(UTC)
            with self.get_session() as session:
                for fact_id in fact_ids:
                    fact = session.get(Fact, fact_id)
                    if fact:
                        fact.notified_at = now
                        session.add(fact)
                session.commit()
                logger.debug("Marked %d facts as notified", len(fact_ids))
        except Exception as e:
            logger.error("Failed to mark facts as notified: %s", e)

    # --- Engagement methods ---

    def add_engagement(
        self,
        user: str,
        engagement_type: str,
        valence: str,
        strength: float,
        entity_id: int | None = None,
        source_message_id: int | None = None,
    ) -> Engagement | None:
        """
        Record a user engagement event.

        Args:
            user: User identifier
            engagement_type: EngagementType enum value
            valence: EngagementValence enum value
            strength: Engagement weight (0.0-1.0)
            entity_id: Optional entity FK
            source_message_id: Optional source message FK

        Returns:
            The created Engagement, or None on failure
        """
        try:
            with self.get_session() as session:
                engagement = Engagement(
                    user=user,
                    entity_id=entity_id,
                    engagement_type=engagement_type,
                    valence=valence,
                    strength=strength,
                    source_message_id=source_message_id,
                )
                session.add(engagement)
                session.commit()
                session.refresh(engagement)
                logger.debug(
                    "Added %s engagement (valence=%s, strength=%.2f) for user %s",
                    engagement_type,
                    valence,
                    strength,
                    user,
                )
                return engagement
        except Exception as e:
            logger.error("Failed to add engagement: %s", e)
            return None

    def get_entity_engagements(self, user: str, entity_id: int) -> list[Engagement]:
        """
        Get all engagements for a specific entity.

        Args:
            user: User identifier
            entity_id: Entity primary key

        Returns:
            List of engagements ordered by created_at descending (newest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Engagement)
                    .where(
                        Engagement.user == user,
                        Engagement.entity_id == entity_id,
                    )
                    .order_by(Engagement.created_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def get_user_engagements(self, user: str) -> list[Engagement]:
        """
        Get all engagements for a user.

        Args:
            user: User identifier

        Returns:
            List of engagements ordered by created_at descending (newest first)
        """
        with self.get_session() as session:
            return list(
                session.exec(
                    select(Engagement)
                    .where(Engagement.user == user)
                    .order_by(Engagement.created_at.desc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    # ── Mute State ────────────────────────────────────────────────────────

    def is_muted(self, user: str) -> bool:
        """Check if a user has muted proactive notifications."""
        with self.get_session() as session:
            return session.get(MuteState, user) is not None

    def set_muted(self, user: str) -> None:
        """Mute proactive notifications for a user."""
        with self.get_session() as session:
            existing = session.get(MuteState, user)
            if not existing:
                session.add(MuteState(user=user))
                session.commit()

    def set_unmuted(self, user: str) -> None:
        """Unmute proactive notifications for a user."""
        with self.get_session() as session:
            existing = session.get(MuteState, user)
            if existing:
                session.delete(existing)
                session.commit()
