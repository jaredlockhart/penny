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
