"""Tests for BrowserChannel message extraction and device registration."""

import asyncio
from unittest.mock import MagicMock

import pytest

from penny.channels.browser.channel import BrowserChannel
from penny.constants import ChannelType
from penny.database import Database
from penny.database.migrate import migrate
from penny.tools.browse_url import BrowseUrlTool


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


class TestBrowserCleanupConnection:
    """_cleanup_connection only rejects pending requests for the disconnected WebSocket."""

    @pytest.mark.asyncio
    async def test_only_rejects_requests_for_disconnected_ws(self, tmp_path):
        db = _make_db(tmp_path)
        channel = BrowserChannel(host="localhost", port=9999, message_agent=MagicMock(), db=db)

        ws_a = MagicMock()
        ws_b = MagicMock()

        loop = asyncio.get_event_loop()
        future_a: asyncio.Future[str] = loop.create_future()
        future_b: asyncio.Future[str] = loop.create_future()

        channel._pending_requests["req-a"] = future_a
        channel._pending_request_connections["req-a"] = ws_a
        channel._pending_requests["req-b"] = future_b
        channel._pending_request_connections["req-b"] = ws_b

        channel._cleanup_connection(ws_a, "device-a")

        assert future_a.done()
        assert isinstance(future_a.exception(), ConnectionError)
        assert not future_b.done()

        future_b.cancel()


class TestBrowseUrlToolConnectionError:
    """BrowseUrlTool returns a friendly message when the browser disconnects."""

    @pytest.mark.asyncio
    async def test_connection_error_returns_friendly_message(self):
        async def disconnected_fn(tool: str, args: dict) -> str:
            raise ConnectionError("Browser disconnected")

        tool = BrowseUrlTool(request_fn=disconnected_fn)
        result = await tool.execute(url="https://example.com")

        assert "not connected" in result
        assert "https://example.com" in result
