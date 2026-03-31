"""Tests for BrowserChannel message extraction and device registration."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest

from penny.channels.browser.channel import BrowserChannel, ConnectionInfo
from penny.constants import ChannelType
from penny.database import Database
from penny.database.migrate import migrate
from penny.tools.browse_url import BrowseUrlTool
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search import SearchTool
from penny.tools.search_emails import SearchEmailsTool


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
    async def test_returns_channel_content_as_search_result(self):
        """Tool returns a SearchResult with the channel content."""
        from unittest.mock import AsyncMock

        from penny.tools.browse_url import BrowseUrlTool
        from penny.tools.models import SearchResult

        request_fn = AsyncMock(
            return_value=("Title: Example\nURL: https://example.com\n\nPage content.", None)
        )
        tool = BrowseUrlTool(request_fn=request_fn)
        result = await tool.execute(url="https://example.com")

        assert isinstance(result, SearchResult)
        assert "Page content." in result.text
        request_fn.assert_called_once_with(BrowseUrlTool.name, {"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_image_url_from_response(self):
        """Tool passes through image URL from the tool response tuple."""
        from unittest.mock import AsyncMock

        from penny.tools.browse_url import BrowseUrlTool
        from penny.tools.models import SearchResult

        request_fn = AsyncMock(
            return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", "https://ex.com/og.jpg")
        )
        tool = BrowseUrlTool(request_fn=request_fn)
        result = await tool.execute(url="https://example.com")

        assert isinstance(result, SearchResult)
        assert result.image_base64 == "https://ex.com/og.jpg"

    @pytest.mark.asyncio
    async def test_no_image_returns_none(self):
        """SearchResult.image_base64 is None when response has no image."""
        from unittest.mock import AsyncMock

        from penny.tools.browse_url import BrowseUrlTool
        from penny.tools.models import SearchResult

        request_fn = AsyncMock(return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", None))
        tool = BrowseUrlTool(request_fn=request_fn)
        result = await tool.execute(url="https://example.com")

        assert isinstance(result, SearchResult)
        assert result.image_base64 is None

    @pytest.mark.asyncio
    async def test_returns_no_content_message_for_empty(self):
        """Tool returns a SearchResult with no-content message when channel returns empty."""
        from unittest.mock import AsyncMock

        from penny.tools.browse_url import BrowseUrlTool
        from penny.tools.models import SearchResult

        request_fn = AsyncMock(return_value=("  ", None))
        tool = BrowseUrlTool(request_fn=request_fn)
        result = await tool.execute(url="https://example.com")

        assert isinstance(result, SearchResult)
        assert "no content" in result.text.lower()

    @pytest.mark.asyncio
    async def test_checks_permission_before_browsing(self):
        """Tool calls permission_manager.check_domain before requesting the page."""
        from unittest.mock import AsyncMock, MagicMock

        from penny.tools.browse_url import BrowseUrlTool

        mock_perm = MagicMock()
        mock_perm.check_domain = AsyncMock()
        request_fn = AsyncMock(return_value=("Title: Ex\nURL: https://ex.com\n\nContent.", None))
        tool = BrowseUrlTool(request_fn=request_fn, permission_manager=mock_perm)
        await tool.execute(url="https://example.com")

        mock_perm.check_domain.assert_called_once_with("https://example.com")
        request_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_permission_denied_raises_before_browse(self):
        """Tool raises without browsing when permission is denied."""
        from unittest.mock import AsyncMock, MagicMock

        from penny.tools.browse_url import BrowseUrlTool

        mock_perm = MagicMock()
        mock_perm.check_domain = AsyncMock(side_effect=RuntimeError("blocked"))
        request_fn = AsyncMock()
        tool = BrowseUrlTool(request_fn=request_fn, permission_manager=mock_perm)

        with pytest.raises(RuntimeError, match="blocked"):
            await tool.execute(url="https://blocked.com")

        request_fn.assert_not_called()


class TestMultiToolImagePassthrough:
    """MultiTool passes the first browse_url image through to the combined result."""

    @pytest.mark.asyncio
    async def test_image_from_browse_url_propagates(self):
        """Image from a browse_url sub-call appears on the combined SearchResult."""
        from unittest.mock import AsyncMock

        from penny.tools.models import SearchResult
        from penny.tools.multi import MultiTool

        browse_result = SearchResult(
            text="Title: Ex\nURL: https://ex.com\nImage: https://ex.com/img.jpg\n\nContent.",
            image_base64="https://ex.com/img.jpg",
        )
        mock_browse_tool = AsyncMock()
        mock_browse_tool.execute = AsyncMock(return_value=browse_result)

        tool = MultiTool(search_tool=None)
        tool.set_browse_url_provider(lambda: mock_browse_tool)

        result = await tool.execute(queries=["https://ex.com"])
        assert isinstance(result, SearchResult)
        assert result.image_base64 == "https://ex.com/img.jpg"

    @pytest.mark.asyncio
    async def test_no_image_when_browse_has_none(self):
        """Combined SearchResult has no image when browse_url returns none."""
        from unittest.mock import AsyncMock

        from penny.tools.models import SearchResult
        from penny.tools.multi import MultiTool

        browse_result = SearchResult(text="Title: Ex\nURL: https://ex.com\n\nContent.")
        mock_browse_tool = AsyncMock()
        mock_browse_tool.execute = AsyncMock(return_value=browse_result)

        tool = MultiTool(search_tool=None)
        tool.set_browse_url_provider(lambda: mock_browse_tool)

        result = await tool.execute(queries=["https://ex.com"])
        assert isinstance(result, SearchResult)
        assert result.image_base64 is None


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


class TestBrowserConfigHandlers:
    """config_request and config_update handlers send and persist correctly."""

    def _channel(self, tmp_path) -> tuple[BrowserChannel, Database]:
        from unittest.mock import MagicMock

        from penny.config_params import RuntimeParams

        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        # Give channel a real RuntimeParams so DB lookups work after updates
        config = MagicMock()
        config.runtime = RuntimeParams(db=db)
        channel._config = config
        return channel, db

    @pytest.mark.asyncio
    async def test_config_request_returns_all_params(self, tmp_path):
        """config_request sends a config_response containing every registered param."""
        from penny.config_params import RUNTIME_CONFIG_PARAMS

        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_request(ws)  # ty: ignore[invalid-argument-type]

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "config_response"
        keys = {p["key"] for p in resp["params"]}
        assert keys == set(RUNTIME_CONFIG_PARAMS.keys())

    @pytest.mark.asyncio
    async def test_config_request_param_shape(self, tmp_path):
        """Each param includes key, value, default, description, type, and group."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_request(ws)  # ty: ignore[invalid-argument-type]

        param = next(p for p in ws.sent[0]["params"] if p["key"] == "IDLE_SECONDS")
        assert param["value"] == "60.0"
        assert param["default"] == "60.0"
        assert param["type"] == "float"
        assert "idle" in param["description"].lower()
        assert param["group"] == "Schedule"

    @pytest.mark.asyncio
    async def test_config_update_persists_value(self, tmp_path):
        """config_update writes the validated value to the runtime_config table."""
        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        channel, db = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "MESSAGE_MAX_STEPS", "value": "12"},
        )

        with Session(db.engine) as session:
            row = session.exec(
                select(RuntimeConfig).where(RuntimeConfig.key == "MESSAGE_MAX_STEPS")
            ).first()
        assert row is not None
        assert row.value == "12"

    @pytest.mark.asyncio
    async def test_config_update_returns_updated_config_response(self, tmp_path):
        """config_update sends back a config_response reflecting the new value."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "MESSAGE_MAX_STEPS", "value": "15"},
        )

        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "config_response"
        param = next(p for p in resp["params"] if p["key"] == "MESSAGE_MAX_STEPS")
        assert param["value"] == "15"

    @pytest.mark.asyncio
    async def test_config_update_unknown_key_is_noop(self, tmp_path):
        """Unknown config key sends nothing and writes nothing to the DB."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "NOT_A_REAL_KEY", "value": "42"},
        )

        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_config_update_invalid_value_is_noop(self, tmp_path):
        """Value that fails validation sends nothing and writes nothing to the DB."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._handle_config_update(
            ws,  # ty: ignore[invalid-argument-type]
            {"type": "config_update", "key": "MESSAGE_MAX_STEPS", "value": "-5"},
        )

        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_config_request_dispatched_via_process_raw_message(self, tmp_path):
        """config_request type is dispatched through _process_raw_message."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "config_request"}),
            None,
        )

        assert len(ws.sent) == 1
        assert ws.sent[0]["type"] == "config_response"

    @pytest.mark.asyncio
    async def test_config_update_dispatched_via_process_raw_message(self, tmp_path):
        """config_update type is dispatched through _process_raw_message."""
        channel, _ = self._channel(tmp_path)
        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "config_update", "key": "MESSAGE_MAX_STEPS", "value": "10"}),
            None,
        )

        assert len(ws.sent) == 1
        assert ws.sent[0]["type"] == "config_response"


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


class TestBrowserRegister:
    """Register message populates _connections so tool requests can be routed."""

    @pytest.mark.asyncio
    async def test_register_populates_connections(self, tmp_path):
        """After register, _connections has the device."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        assert len(channel._connections) == 0

        ws = _MockWs()
        label = await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            None,
        )

        assert label == "firefox-macbook"
        assert "firefox-macbook" in channel._connections
        assert channel._connections["firefox-macbook"].ws is ws

    @pytest.mark.asyncio
    async def test_register_creates_device_in_db(self, tmp_path):
        """Register auto-registers the device in the database."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            None,
        )

        device = db.devices.get_by_identifier("firefox-macbook")
        assert device is not None
        assert device.label == "firefox-macbook"

    @pytest.mark.asyncio
    async def test_tool_request_works_after_register_without_chat(self, tmp_path):
        """Tool requests succeed after register + capabilities even if no chat message was sent."""
        import asyncio

        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-macbook"}),
            None,
        )
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "capabilities_update", "tool_use_enabled": True}),
            "firefox-macbook",
        )

        # Pre-allow the domain so the permission check passes
        db.domain_permissions.set_permission("example.com", "allowed")

        # Simulate a tool response arriving after we send the request
        async def fake_tool_response():
            await asyncio.sleep(0.05)
            # Find the pending request and resolve it
            for _req_id, future in channel._pending_requests.items():
                if not future.done():
                    future.set_result(("page content here", None))
                    break

        asyncio.create_task(fake_tool_response())
        result = await channel.send_tool_request("browse_url", {"url": "https://example.com"})
        assert result == ("page content here", None)


class TestCapabilitiesAndToolRouting:
    """Tool-use toggle and smart routing based on capabilities."""

    async def _register(self, channel, label, ws=None):
        """Register a browser connection by device label."""
        ws = ws or _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": label}),
            None,
        )
        return ws

    async def _set_capabilities(self, channel, label, ws, tool_use_enabled):
        """Send a capabilities_update for a registered connection."""
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "capabilities_update", "tool_use_enabled": tool_use_enabled}),
            label,
        )

    @pytest.mark.asyncio
    async def test_capabilities_update_sets_tool_use(self, tmp_path):
        """capabilities_update toggles tool_use_enabled on the connection."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = await self._register(channel, "firefox-1")
        assert not channel._connections["firefox-1"].tool_use_enabled

        await self._set_capabilities(channel, "firefox-1", ws, True)
        assert channel._connections["firefox-1"].tool_use_enabled

        await self._set_capabilities(channel, "firefox-1", ws, False)
        assert not channel._connections["firefox-1"].tool_use_enabled

    @pytest.mark.asyncio
    async def test_has_tool_connection_requires_tool_use_enabled(self, tmp_path):
        """has_tool_connection is False when connections exist but none have tool_use enabled."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws = await self._register(channel, "firefox-1")
        assert not channel.has_tool_connection

        await self._set_capabilities(channel, "firefox-1", ws, True)
        assert channel.has_tool_connection

    @pytest.mark.asyncio
    async def test_get_tool_connection_picks_enabled_addon(self, tmp_path):
        """Smart routing picks the tool-enabled connection, not the first one."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        await self._register(channel, "firefox-personal")
        ws_penny = await self._register(channel, "firefox-penny")

        # Only enable tool use on the second one
        await self._set_capabilities(channel, "firefox-penny", ws_penny, True)

        routed = channel._get_tool_connection()
        assert routed is ws_penny

    @pytest.mark.asyncio
    async def test_get_tool_connection_none_when_all_disabled(self, tmp_path):
        """Returns None when no connections have tool_use enabled."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        await self._register(channel, "firefox-1")
        await self._register(channel, "firefox-2")

        assert channel._get_tool_connection() is None


class TestBrowserPermissionDelegation:
    """BrowserChannel delegates permission checks to PermissionManager."""

    async def _setup_channel(self, tmp_path):
        """Create a channel with a registered, tool-enabled connection and permission manager."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        ws = _MockWs()
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-penny"}),
            None,
        )
        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "capabilities_update", "tool_use_enabled": True}),
            "firefox-penny",
        )
        return channel, db, ws

    @pytest.mark.asyncio
    async def test_permission_decision_routes_to_manager(self, tmp_path):
        """permission_decision message routes to the permission manager."""
        channel, db, ws = await self._setup_channel(tmp_path)
        mock_perm_mgr = MagicMock()
        channel.set_permission_manager(mock_perm_mgr)

        await channel._process_raw_message(
            ws,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "permission_decision", "request_id": "test-123", "allowed": True}),
            "firefox-penny",
        )

        mock_perm_mgr.handle_decision.assert_called_once_with("test-123", True)

    @pytest.mark.asyncio
    async def test_handle_permission_prompt_sends_to_all_addons(self, tmp_path):
        """handle_permission_prompt sends prompt to all connected addons."""
        channel, db, ws1 = await self._setup_channel(tmp_path)

        ws2 = _MockWs()
        await channel._process_raw_message(
            ws2,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-personal"}),
            None,
        )

        await channel.handle_permission_prompt("req-1", "example.com", "https://example.com/")

        for ws in [ws1, ws2]:
            prompts = [m for m in ws.sent if m.get("type") == "permission_prompt"]
            assert len(prompts) == 1
            assert prompts[0]["domain"] == "example.com"

    @pytest.mark.asyncio
    async def test_handle_permission_dismiss_sends_to_all_addons(self, tmp_path):
        """handle_permission_dismiss sends dismiss to all connected addons."""
        channel, db, ws1 = await self._setup_channel(tmp_path)

        ws2 = _MockWs()
        await channel._process_raw_message(
            ws2,  # ty: ignore[invalid-argument-type]
            json.dumps({"type": "register", "sender": "firefox-personal"}),
            None,
        )

        await channel.handle_permission_dismiss("req-1")

        for ws in [ws1, ws2]:
            dismissals = [m for m in ws.sent if m.get("type") == "permission_dismiss"]
            assert len(dismissals) == 1


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


class TestFormatToolStatus:
    """_format_tool_status produces human-readable labels for each tool."""

    def test_search_single_query(self):
        result = BrowserChannel._format_tool_status(SearchTool.name, {"query": "firefox memory"})
        assert result == 'Searching for "firefox memory"'

    def test_search_invalid_args(self):
        result = BrowserChannel._format_tool_status(SearchTool.name, {})
        assert result == "Searching"

    def test_browse_url_with_url(self):
        result = BrowserChannel._format_tool_status(
            BrowseUrlTool.name, {"url": "https://example.com"}
        )
        assert result == "Reading https://example.com"

    def test_browse_url_without_url(self):
        result = BrowserChannel._format_tool_status(BrowseUrlTool.name, {})
        assert result == "Reading page"

    def test_fetch_news_with_topic(self):
        result = BrowserChannel._format_tool_status(FetchNewsTool.name, {"topic": "climate change"})
        assert result == "Fetching news about climate change"

    def test_fetch_news_default_topic(self):
        result = BrowserChannel._format_tool_status(FetchNewsTool.name, {})
        assert result == "Fetching news about top news"

    def test_search_emails(self):
        result = BrowserChannel._format_tool_status(SearchEmailsTool.name, {"text": "invoice"})
        assert result == "Searching emails"

    def test_read_emails(self):
        result = BrowserChannel._format_tool_status(ReadEmailsTool.name, {"email_ids": ["123"]})
        assert result == "Reading emails"

    def test_unknown_tool(self):
        result = BrowserChannel._format_tool_status("my_custom_tool", {})
        assert result == "Using my_custom_tool"


class TestMakeHandleKwargs:
    """_make_handle_kwargs returns a callback that sends tool status to the browser."""

    @pytest.mark.asyncio
    async def test_returns_on_tool_start_key(self, tmp_path):
        """_make_handle_kwargs always returns a dict with an on_tool_start callable."""
        from penny.channels.base import IncomingMessage

        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        message = IncomingMessage(sender="browser-user", content="hello")
        kwargs = channel._make_handle_kwargs(message)

        assert "on_tool_start" in kwargs
        assert callable(kwargs["on_tool_start"])

    @pytest.mark.asyncio
    async def test_callback_sends_tool_status(self, tmp_path):
        """Callback calls _send_tool_status with the sender and formatted text."""
        from unittest.mock import AsyncMock

        from penny.channels.base import IncomingMessage

        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        channel._send_tool_status = AsyncMock()  # ty: ignore[invalid-assignment]

        message = IncomingMessage(sender="firefox-macbook", content="hello")
        kwargs = channel._make_handle_kwargs(message)
        await kwargs["on_tool_start"]([(SearchTool.name, {"query": "test query"})])

        channel._send_tool_status.assert_called_once()
        recipient, text = channel._send_tool_status.call_args.args
        assert recipient == "firefox-macbook"
        assert '"test query"' in text

    @pytest.mark.asyncio
    async def test_send_tool_status_sends_typing_with_content(self, tmp_path):
        """_send_tool_status sends a typing message with the status text as content."""
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)
        ws = _MockWs()
        cast(dict, channel._connections)["browser-user"] = ConnectionInfo(ws=ws)  # ty: ignore[invalid-argument-type]

        await channel._send_tool_status("browser-user", "Searching for stuff")

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "typing"
        assert msg["active"] is True
        assert msg["content"] == "Searching for stuff"
