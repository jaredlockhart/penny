"""Database facade — composes domain-specific stores."""

import logging
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from penny.database.engagement_store import EngagementStore
from penny.database.entity_store import EntityStore
from penny.database.event_store import EventStore
from penny.database.fact_store import FactStore
from penny.database.follow_prompt_store import FollowPromptStore
from penny.database.learn_prompt_store import LearnPromptStore
from penny.database.message_store import MessageStore
from penny.database.search_store import SearchStore
from penny.database.user_store import UserStore

logger = logging.getLogger(__name__)


class Database:
    """Database facade — provides access to domain-specific stores.

    Stores:
        entities: Entity CRUD, embeddings, taglines, metadata
        events: Event CRUD, entity linking, dedup, notification tracking
        facts: Fact CRUD, embeddings, notification tracking
        follow_prompts: FollowPrompt lifecycle for event monitoring
        messages: Message/prompt/command logging, threading, queries
        learn_prompts: LearnPrompt lifecycle and cascading deletion
        searches: SearchLog creation and extraction tracking
        engagements: Engagement recording and queries
        users: UserInfo, sender queries, mute state
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")

        self.entities = EntityStore(self.engine)
        self.events = EventStore(self.engine)
        self.facts = FactStore(self.engine)
        self.follow_prompts = FollowPromptStore(self.engine)
        self.messages = MessageStore(self.engine)
        self.learn_prompts = LearnPromptStore(self.engine)
        self.searches = SearchStore(self.engine)
        self.engagements = EngagementStore(self.engine)
        self.users = UserStore(self.engine)

        logger.info("Database initialized: %s", db_path)

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Session:
        """Get a database session (for direct use by schedule/config modules)."""
        return Session(self.engine)
