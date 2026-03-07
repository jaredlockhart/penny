"""Database facade — composes domain-specific stores."""

import logging
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from penny.database.history_store import HistoryStore
from penny.database.message_store import MessageStore
from penny.database.preference_store import PreferenceStore
from penny.database.search_store import SearchStore
from penny.database.thought_store import ThoughtStore
from penny.database.user_store import UserStore

logger = logging.getLogger(__name__)


class Database:
    """Database facade — provides access to domain-specific stores.

    Stores:
        history: Conversation topic summaries for long-term context
        messages: Message/prompt/command logging, threading, queries
        preferences: User preference CRUD and dedup
        searches: SearchLog creation and extraction tracking
        thoughts: Inner monologue persistence (append-only thought log)
        users: UserInfo, sender queries, mute state
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")

        self.history = HistoryStore(self.engine)
        self.messages = MessageStore(self.engine)
        self.preferences = PreferenceStore(self.engine)
        self.searches = SearchStore(self.engine)
        self.thoughts = ThoughtStore(self.engine)
        self.users = UserStore(self.engine)

        logger.info("Database initialized: %s", db_path)

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Session:
        """Get a database session (for direct use by schedule/config modules)."""
        return Session(self.engine)
