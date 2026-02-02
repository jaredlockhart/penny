"""SQLModel models for Penny's memory."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    """Message log - tracks all incoming and outgoing messages."""

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    direction: str = Field(index=True)  # "incoming" or "outgoing"
    sender: str = Field(index=True)  # Phone number
    recipient: str = Field(index=True)  # Phone number
    content: str
    chunk_index: Optional[int] = Field(default=None)  # For streaming chunks
    thinking: Optional[str] = Field(default=None)  # LLM reasoning for thinking models


class Memory(SQLModel, table=True):
    """Long-term memory storage for facts, preferences, and rules."""

    id: Optional[int] = Field(default=None, primary_key=True)
    content: str = Field(index=True)  # The memory text
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class Task(SQLModel, table=True):
    """Deferred task that Penny will work on during idle time."""

    id: Optional[int] = Field(default=None, primary_key=True)
    content: str = Field(index=True)  # The task description
    status: str = Field(default="pending", index=True)  # pending/in_progress/completed
    requester: str = Field(index=True)  # Phone number of requester
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None  # Final result text
