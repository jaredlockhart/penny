"""Tests for BrowserChannel message extraction and device registration."""

from unittest.mock import MagicMock

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
