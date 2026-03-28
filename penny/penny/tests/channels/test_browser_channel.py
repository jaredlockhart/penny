"""Tests for BrowserChannel message extraction and device registration."""

import json
from unittest.mock import MagicMock

import pytest

from penny.channels.browser.channel import BrowserChannel
from penny.constants import ChannelType
from penny.database import Database
from penny.database.migrate import migrate


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


class TestBrowserChannelExtract:
    """extract_message produces IncomingMessage with correct fields."""

    def test_extracts_message_with_channel_type(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        raw = {"browser_sender": "firefox-macbook", "content": "hello penny"}
        msg = channel.extract_message(raw)

        assert msg is not None
        assert msg.sender == "firefox-macbook"
        assert msg.content == "hello penny"
        assert msg.channel_type == ChannelType.BROWSER
        assert msg.device_identifier == "firefox-macbook"

    def test_extracts_default_sender(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        raw = {"content": "hello"}
        msg = channel.extract_message(raw)

        assert msg is not None
        assert msg.sender == "browser-user"
        assert msg.device_identifier == "browser-user"

    def test_returns_none_for_empty_content(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        assert channel.extract_message({"content": ""}) is None
        assert channel.extract_message({"content": "   "}) is None


class TestBrowserAutoRegistration:
    """_auto_register_device creates device entries in the database."""

    def test_registers_new_device(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        channel._auto_register_device("firefox-macbook-16")

        device = db.devices.get_by_identifier("firefox-macbook-16")
        assert device is not None
        assert device.channel_type == ChannelType.BROWSER
        assert device.label == "firefox-macbook-16"

    def test_register_is_idempotent(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        channel._auto_register_device("firefox-macbook-16")
        channel._auto_register_device("firefox-macbook-16")

        all_devices = db.devices.get_all()
        browser_devices = [d for d in all_devices if d.identifier == "firefox-macbook-16"]
        assert len(browser_devices) == 1


class TestBrowserPrepareOutgoing:
    """prepare_outgoing converts markdown to HTML."""

    def _channel(self, tmp_path):
        db = _make_db(tmp_path)
        return BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

    def test_bold(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("**hello**")
        assert "<strong>hello</strong>" in result

    def test_italic(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("*hello*")
        assert "<em>hello</em>" in result

    def test_strikethrough(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("~~deleted~~")
        assert "<s>deleted</s>" in result

    def test_inline_code(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("use `pip install`")
        assert "<code>pip install</code>" in result

    def test_fenced_code_block(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("```\nprint('hi')\n```")
        assert "<pre><code>" in result
        assert "print" in result

    def test_heading_becomes_strong(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("## Section Title")
        assert "<strong>Section Title</strong>" in result

    def test_markdown_link(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("[click](https://example.com)")
        assert '<a href="https://example.com"' in result
        assert "click</a>" in result

    def test_bare_url(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("visit https://example.com today")
        assert '<a href="https://example.com"' in result

    def test_markdown_link_no_nested_anchors(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("[Recipe](https://example.com/recipe)")
        assert result.count("<a ") == 1
        assert result.count("</a>") == 1
        assert '<a href="https://example.com/recipe" target="_blank">Recipe</a>' in result

    def test_html_escaped(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("use <script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_table_to_bullets(self, tmp_path):
        table = "| Model | Price |\n|-------|-------|\n| Foo   | $100  |\n| Bar   | $200  |"
        result = self._channel(tmp_path).prepare_outgoing(table)
        assert "<strong>Foo</strong>" in result
        assert "$100" in result
        assert "<strong>Bar</strong>" in result
        assert "|" not in result

    def test_newlines_to_br(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("line one\nline two")
        assert "<br>" in result

    def test_collapses_excessive_breaks(self, tmp_path):
        result = self._channel(tmp_path).prepare_outgoing("a\n\n\n\n\nb")
        assert "<br><br><br>" not in result


class TestBrowserImageHandling:
    """_prepend_images puts images before the message content."""

    def test_prepends_image_url(self):
        result = BrowserChannel._prepend_images("hello", ["https://example.com/img.jpg"])
        assert result.startswith('<img src="https://example.com/img.jpg"')
        assert result.endswith("hello")

    def test_prepends_data_uri(self):
        result = BrowserChannel._prepend_images("hello", ["data:image/png;base64,abc123"])
        assert '<img src="data:image/png;base64,abc123"' in result
        assert result.endswith("hello")

    def test_prepends_raw_base64_as_data_uri(self):
        """Raw base64 from /draw gets wrapped in a data:image/png URI."""
        raw_b64 = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200
        result = BrowserChannel._prepend_images("hello", [raw_b64])
        assert '<img src="data:image/png;base64,' in result
        assert result.endswith("hello")

    def test_skips_short_non_url(self):
        result = BrowserChannel._prepend_images("hello", ["short"])
        assert result == "hello"

    def test_no_attachments(self):
        assert BrowserChannel._prepend_images("hello", None) == "hello"
        assert BrowserChannel._prepend_images("hello", []) == "hello"

    def test_multiple_images(self):
        urls = ["https://example.com/a.jpg", "https://example.com/b.jpg"]
        result = BrowserChannel._prepend_images("text", urls)
        assert result.count("<img") == 2
        assert result.endswith("text")


class TestBrowseUrlTool:
    """BrowseUrlTool passes through pre-sanitized content from the channel."""

    @pytest.mark.asyncio
    async def test_returns_channel_content_directly(self):
        """Tool returns whatever the channel's request_fn provides — no summarization."""
        from unittest.mock import AsyncMock

        from penny.tools.browse_url import BrowseUrlTool

        request_fn = AsyncMock(return_value="Pre-sanitized page content from channel.")
        tool = BrowseUrlTool(request_fn=request_fn)
        result = await tool.execute(url="https://example.com")

        assert result == "Pre-sanitized page content from channel."
        request_fn.assert_called_once_with("browse_url", {"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_returns_no_content_message_for_empty(self):
        """Tool returns a message when the channel returns empty content."""
        from unittest.mock import AsyncMock

        from penny.tools.browse_url import BrowseUrlTool

        request_fn = AsyncMock(return_value="  ")
        tool = BrowseUrlTool(request_fn=request_fn)
        result = await tool.execute(url="https://example.com")

        assert "no content" in result.lower()


class _MockWs:
    """Minimal mock WebSocket that captures sent JSON messages."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))


class TestBrowserPreferenceHandlers:
    """Preference request/add/delete handlers send correct WebSocket responses."""

    USER = "testuser"

    def _channel(self, tmp_path, monkeypatch) -> tuple[BrowserChannel, Database]:
        db = _make_db(tmp_path)
        monkeypatch.setattr(db.users, "get_primary_sender", lambda: self.USER)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        return channel, db

    @pytest.mark.asyncio
    async def test_preferences_request_empty(self, tmp_path, monkeypatch):
        """Request with no preferences sends an empty list."""
        channel, _ = self._channel(tmp_path, monkeypatch)
        ws = _MockWs()
        await channel._handle_preferences_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "preferences_request", "valence": "positive"},
        )

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "preferences_response"
        assert resp["valence"] == "positive"
        assert resp["preferences"] == []

    @pytest.mark.asyncio
    async def test_preferences_request_filters_by_valence(self, tmp_path, monkeypatch):
        """Only preferences matching the requested valence are returned."""
        channel, db = self._channel(tmp_path, monkeypatch)
        db.preferences.add(self.USER, "dark roast coffee", "positive", source="manual")
        db.preferences.add(self.USER, "hiking", "positive", source="manual")
        db.preferences.add(self.USER, "cold weather", "negative", source="manual")

        ws = _MockWs()
        await channel._handle_preferences_request(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "preferences_request", "valence": "positive"},
        )

        resp = ws.sent[0]
        contents = [p["content"] for p in resp["preferences"]]
        assert "dark roast coffee" in contents
        assert "hiking" in contents
        assert "cold weather" not in contents

    @pytest.mark.asyncio
    async def test_preference_add_stores_and_returns_list(self, tmp_path, monkeypatch):
        """preference_add persists the preference as manual and returns the updated list."""
        channel, db = self._channel(tmp_path, monkeypatch)
        ws = _MockWs()
        await channel._handle_preference_add(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "preference_add", "valence": "positive", "content": "jazz music"},
        )

        resp = ws.sent[0]
        assert resp["type"] == "preferences_response"
        assert resp["valence"] == "positive"
        assert resp["preferences"][0]["content"] == "jazz music"

        saved = db.preferences.get_for_user_by_valence(self.USER, "positive")
        assert len(saved) == 1
        assert saved[0].source == "manual"

    @pytest.mark.asyncio
    async def test_preference_delete_removes_and_returns_list(self, tmp_path, monkeypatch):
        """preference_delete removes the entry and returns the remaining list."""
        channel, db = self._channel(tmp_path, monkeypatch)
        pref = db.preferences.add(self.USER, "jazz music", "positive", source="manual")
        assert pref is not None
        db.preferences.add(self.USER, "hiking", "positive", source="manual")

        ws = _MockWs()
        await channel._handle_preference_delete(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "preference_delete", "preference_id": pref.id},
        )

        resp = ws.sent[0]
        remaining = [p["content"] for p in resp["preferences"]]
        assert "jazz music" not in remaining
        assert "hiking" in remaining

    @pytest.mark.asyncio
    async def test_preference_add_ignores_blank_content(self, tmp_path, monkeypatch):
        """preference_add with whitespace-only content sends nothing and stores nothing."""
        channel, db = self._channel(tmp_path, monkeypatch)
        ws = _MockWs()
        await channel._handle_preference_add(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "preference_add", "valence": "positive", "content": "   "},
        )

        assert ws.sent == []
        assert db.preferences.get_for_user_by_valence(self.USER, "positive") == []

    @pytest.mark.asyncio
    async def test_preference_delete_unknown_id_is_noop(self, tmp_path, monkeypatch):
        """preference_delete with an unknown ID sends nothing."""
        channel, _ = self._channel(tmp_path, monkeypatch)
        ws = _MockWs()
        await channel._handle_preference_delete(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "preference_delete", "preference_id": 9999},
        )

        assert ws.sent == []


class TestBrowserHeartbeat:
    """Heartbeat resets the scheduler idle timer without touching schedule intervals."""

    @pytest.mark.asyncio
    async def test_heartbeat_calls_notify_activity(self, tmp_path):
        from unittest.mock import MagicMock

        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        scheduler = MagicMock()
        channel.set_scheduler(scheduler)

        ws = _MockWs()
        await channel._process_raw_message(ws, '{"type": "heartbeat"}', None)  # ty: ignore[invalid-argument-type]

        scheduler.notify_activity.assert_called_once()
        scheduler.notify_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_without_scheduler_is_noop(self, tmp_path):
        """No scheduler set — heartbeat is silently ignored."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = _MockWs()
        # Should not raise
        await channel._process_raw_message(ws, '{"type": "heartbeat"}', None)  # ty: ignore[invalid-argument-type]


class TestBrowserThoughtReaction:
    """_handle_thought_reaction stores valence on thought and marks it notified."""

    USER = "testuser"

    def _setup(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path)
        monkeypatch.setattr(db.users, "get_primary_sender", lambda: self.USER)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        thought = db.thoughts.add(self.USER, "Some fascinating thought content")
        assert thought is not None
        return channel, db, thought

    def test_positive_emoji_sets_valence_and_marks_notified(self, tmp_path, monkeypatch):
        channel, db, thought = self._setup(tmp_path, monkeypatch)
        channel._handle_thought_reaction({"thought_id": thought.id, "emoji": "👍"})

        updated = db.thoughts.get_by_id(thought.id)
        assert updated.valence == 1
        assert updated.notified_at is not None

    def test_negative_emoji_sets_valence_and_marks_notified(self, tmp_path, monkeypatch):
        channel, db, thought = self._setup(tmp_path, monkeypatch)
        channel._handle_thought_reaction({"thought_id": thought.id, "emoji": "👎"})

        updated = db.thoughts.get_by_id(thought.id)
        assert updated.valence == -1
        assert updated.notified_at is not None

    def test_unknown_emoji_no_valence_set(self, tmp_path, monkeypatch):
        channel, db, thought = self._setup(tmp_path, monkeypatch)
        channel._handle_thought_reaction({"thought_id": thought.id, "emoji": "🐱"})

        updated = db.thoughts.get_by_id(thought.id)
        assert updated.valence is None
        assert updated.notified_at is not None  # still marked notified

    def test_no_synthetic_messages_created(self, tmp_path, monkeypatch):
        channel, db, thought = self._setup(tmp_path, monkeypatch)
        channel._handle_thought_reaction({"thought_id": thought.id, "emoji": "👍"})

        reactions = db.messages.get_user_reactions(self.USER, limit=100)
        assert reactions == []
