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
