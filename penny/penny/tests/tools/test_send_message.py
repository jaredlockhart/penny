"""Tests for SendMessageTool — mute and cooldown gates.

The tool is the universal outbound delivery primitive.  Two gates
are enforced before the channel dispatch:

1. ``users.is_muted(recipient)`` — refuses with a string that
   instructs the model to call ``done``.
2. Exponential backoff cooldown based on prior sends from this
   agent in ``penny-messages`` since the user's last entry in
   ``user-messages``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import LogEntryInput, RecallMode
from penny.tools.send_message import SendMessageTool

_PENNY_LOG = PennyConstants.MEMORY_PENNY_MESSAGES_LOG
_USER_LOG = PennyConstants.MEMORY_USER_MESSAGES_LOG

_RECIPIENT = "+15551234567"
_AGENT = "notify"


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    # The cooldown helper reads the system penny-messages and user-messages
    # logs; create them up-front so the tool's lookups don't ImportError.
    db.memories.create_log(_PENNY_LOG, "outbound", RecallMode.OFF)
    db.memories.create_log(_USER_LOG, "inbound", RecallMode.OFF)
    return db


def _make_config(min_cooldown: float = 60.0, max_cooldown: float = 3600.0):
    """Stand-in config exposing the runtime knobs the tool reads."""
    runtime = type(
        "Runtime",
        (),
        {"NOTIFY_COOLDOWN_MIN": min_cooldown, "NOTIFY_COOLDOWN_MAX": max_cooldown},
    )()
    return type("Config", (), {"runtime": runtime})()


def _make_channel():
    """Mock channel — only ``send_response`` is exercised."""
    channel = type("Channel", (), {})()
    channel.send_response = AsyncMock(return_value=42)
    return channel


def _make_tool(db, channel=None, config=None):
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )
    return SendMessageTool(
        channel=channel or _make_channel(),
        agent_name=_AGENT,
        db=db,
        config=config or _make_config(),
    )


@pytest.mark.asyncio
async def test_send_message_dispatches_when_not_gated(tmp_path):
    """Happy path: no mute, no prior sends → dispatch + ack."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(content="hey there!")

    assert result == "Message sent."
    channel.send_response.assert_awaited_once()
    kwargs = channel.send_response.await_args.kwargs
    assert kwargs["recipient"] == _RECIPIENT
    assert kwargs["content"] == "hey there!"
    assert kwargs["author"] == _AGENT


@pytest.mark.asyncio
async def test_send_message_refuses_when_user_muted(tmp_path):
    """Muted recipient: tool refuses without dispatching."""
    db = _make_db(tmp_path)
    db.users.set_muted(_RECIPIENT)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(content="hey there!")

    assert "muted" in result.lower()
    assert "done" in result.lower()
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_message_refuses_when_content_is_a_refusal(tmp_path):
    """Refusal content ("I'm sorry, I can't...") is not dispatched as a reply."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel)

    result = await tool.execute(
        content="I'm sorry, I can't help with that as an AI language model."
    )

    assert "refusal" in result.lower()
    assert "done" in result.lower()
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_message_refuses_when_cooldown_not_elapsed(tmp_path):
    """A recent send from the same agent (no user reply since) → cooldown gate fires."""
    db = _make_db(tmp_path)
    # Seed a prior send authored by this agent — count = 1, cooldown = MIN.
    db.memories.append(_PENNY_LOG, [LogEntryInput(content="prior")], author=_AGENT)
    channel = _make_channel()
    tool = _make_tool(db, channel=channel, config=_make_config(min_cooldown=3600.0))

    result = await tool.execute(content="hey again!")

    assert "cooldown" in result.lower()
    assert "done" in result.lower()
    channel.send_response.assert_not_awaited()


def test_user_reply_resets_cooldown_count(tmp_path):
    """A user-messages entry newer than prior sends resets the backoff count to zero."""
    db = _make_db(tmp_path)
    # Old send → would normally count toward backoff.
    db.memories.append(_PENNY_LOG, [LogEntryInput(content="old")], author=_AGENT)
    # User replied since — entries are timestamped at write time, so this
    # user-messages entry's created_at is newer than the old send.
    db.memories.append(_USER_LOG, [LogEntryInput(content="hi back")], author="user")
    tool = _make_tool(db)

    # The count walks newest-first and breaks once entries are older than the
    # latest user message — an immediate break gives count = 0.
    assert tool._count_sends_since_user_message() == 0


def test_latest_send_time_filters_by_author(tmp_path):
    """Only entries authored by ``self.agent_name`` count toward this agent's history."""
    db = _make_db(tmp_path)
    db.memories.append(_PENNY_LOG, [LogEntryInput(content="from chat")], author="chat")
    db.memories.append(_PENNY_LOG, [LogEntryInput(content="from notify")], author=_AGENT)
    db.memories.append(_PENNY_LOG, [LogEntryInput(content="another chat")], author="chat")
    tool = _make_tool(db)

    latest = tool._latest_send_time()
    assert latest is not None  # the notify entry is found despite chat entries surrounding it


def test_latest_send_time_none_when_no_prior_sends(tmp_path):
    """Empty log → None, which the cooldown helper treats as 'no cooldown to wait out'."""
    db = _make_db(tmp_path)
    tool = _make_tool(db)
    assert tool._latest_send_time() is None
