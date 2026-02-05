"""Pytest fixtures for Penny tests."""

import pytest

from penny.config import Config
from penny.tests.mocks.ollama_server import MockOllamaServer

# Re-export search patches so they can be used as fixtures
from penny.tests.mocks.search_patches import mock_search, mock_search_with_results  # noqa: F401
from penny.tests.mocks.signal_server import MockSignalServer

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
async def signal_server():
    """Start a mock Signal server and yield it."""
    server = MockSignalServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def ollama_server():
    """Start a mock Ollama server and yield it."""
    server = MockOllamaServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def test_config(signal_server, ollama_server, test_db):
    """
    Create a test Config pointing to mock servers.

    Background schedules are disabled by setting high idle times.
    """
    return Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url=f"http://localhost:{signal_server.port}",
        discord_bot_token=None,
        discord_channel_id=None,
        ollama_api_url=f"http://localhost:{ollama_server.port}",
        ollama_foreground_model="test-model",
        ollama_background_model="test-model",
        perplexity_api_key="test-api-key",
        log_level="DEBUG",
        db_path=test_db,
        # Disable background tasks by setting very long idle times
        summarize_idle_seconds=99999.0,
        profile_idle_seconds=99999.0,
        followup_idle_seconds=99999.0,
        followup_min_seconds=99999.0,
        followup_max_seconds=99999.0,
        discovery_idle_seconds=99999.0,
        discovery_min_seconds=99999.0,
        discovery_max_seconds=99999.0,
        # Fast retries for tests
        ollama_max_retries=1,
        ollama_retry_delay=0.1,
    )
