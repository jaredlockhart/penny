"""Database connection and session management."""

import logging
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger(__name__)


class Database:
    """Database manager for Penny's memory."""

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

    def get_conversation_history(self, sender: str, recipient: str, limit: int = 20) -> list:
        """
        Get recent conversation history between sender and recipient.

        Args:
            sender: Phone number of sender
            recipient: Phone number of recipient
            limit: Maximum number of messages to retrieve

        Returns:
            List of Message objects, ordered by timestamp (oldest first)
        """
        from penny.memory.models import Message

        with self.get_session() as session:
            # Get messages where either party is sender or recipient
            messages = session.query(Message).filter(
                ((Message.sender == sender) & (Message.recipient == recipient))
                | ((Message.sender == recipient) & (Message.recipient == sender))
            ).order_by(Message.timestamp.desc()).limit(limit).all()

            # Reverse to get chronological order (oldest first)
            return list(reversed(messages))
