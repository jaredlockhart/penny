"""Pytest fixtures for Penny tests."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, cast

import pytest

from penny.config import Config
from penny.penny import Penny

# Re-export mock fixtures so they can be used directly in tests
from penny.tests.mocks.ollama_patches import mock_ollama  # noqa: F401
from penny.tests.mocks.search_patches import (
    mock_search,  # noqa: F401
    mock_search_with_results,  # noqa: F401
)
from penny.tests.mocks.search_patches import mock_search as _mock_search  # noqa: F401
from penny.tests.mocks.signal_server import MockSignalServer

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

# Standard test sender phone number
TEST_SENDER = "+15559876543"

# Default config values for tests (background tasks disabled)
DEFAULT_TEST_CONFIG = {
    "channel_type": "signal",
    "signal_number": "+15551234567",
    "discord_bot_token": None,
    "discord_channel_id": None,
    "ollama_api_url": "http://localhost:11434",
    "ollama_foreground_model": "test-model",
    "ollama_background_model": "test-model",
    "perplexity_api_key": "test-api-key",
    "log_level": "DEBUG",
    "tool_timeout": 60.0,
    # Disable background tasks by default
    "idle_seconds": 99999.0,
    "followup_min_seconds": 99999.0,
    "followup_max_seconds": 99999.0,
    "discovery_min_seconds": 99999.0,
    "discovery_max_seconds": 99999.0,
    # Fast scheduler ticks for tests
    "scheduler_tick_interval": 0.05,
    # Fast research schedule interval for tests (prod: 5.0s)
    "research_schedule_interval": 0.1,
    # Fast retries for tests
    "ollama_max_retries": 1,
    "ollama_retry_delay": 0.1,
}


async def wait_until(
    condition: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.05,
) -> None:
    """
    Poll a condition until it becomes true, or raise TimeoutError.

    Replaces arbitrary ``asyncio.sleep(N)`` calls in tests with deterministic,
    condition-based waiting that returns as soon as the expected state is reached.

    Args:
        condition: Synchronous callable that returns True when ready.
        timeout: Maximum seconds to wait before raising TimeoutError.
        interval: Seconds between polls.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"Condition not met within {timeout}s")


@pytest.fixture
async def signal_server():
    """Start a mock Signal server and yield it."""
    server = MockSignalServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def make_config(signal_server, test_db) -> Callable[..., Config]:
    """
    Factory fixture for creating test configs with custom overrides.

    Usage:
        config = make_config()  # defaults
        config = make_config(summarize_idle_seconds=0.5)  # with override
    """

    def _make_config(**overrides: Any) -> Config:
        config_kwargs: dict[str, Any] = {
            **DEFAULT_TEST_CONFIG,
            "signal_api_url": f"http://localhost:{signal_server.port}",
            "db_path": test_db,
            **overrides,
        }
        return Config(**cast(Any, config_kwargs))

    return _make_config


@pytest.fixture
def test_config(make_config) -> Config:
    """
    Create a test Config pointing to mock servers.

    Background schedules are disabled by setting high idle times.
    For custom configs, use make_config fixture instead.
    """
    return make_config()


@pytest.fixture
def test_user_info(test_config):
    """
    Create a test user profile to bypass profile prompting.

    This sets up a UserInfo record for TEST_SENDER so tests don't get
    intercepted by profile collection prompts. The DB is initialized (tables
    created, then migrations run) before creating the user.
    """
    from penny.database import Database
    from penny.database.migrate import migrate

    # Create database and tables first
    db = Database(test_config.db_path)
    db.create_tables()

    # Then run migrations
    migrate(test_config.db_path)

    # Now create the test user
    db.save_user_info(
        sender=TEST_SENDER,
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )
    return db


@pytest.fixture
def running_penny(signal_server) -> Callable[[Config], AbstractAsyncContextManager[Penny]]:
    """
    Async context manager fixture for running Penny with proper cleanup.

    Usage:
        async with running_penny(config) as penny:
            # penny is running and ready
            await signal_server.push_message(...)
    """

    @asynccontextmanager
    async def _running_penny(config: Config) -> AsyncIterator[Penny]:
        penny = Penny(config)
        penny_task = asyncio.create_task(penny.run())
        try:
            # Wait for WebSocket connection to establish
            await wait_until(lambda: len(signal_server._websockets) > 0)
            yield penny
        finally:
            penny_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await penny_task
            await penny.shutdown()

    return _running_penny


@pytest.fixture
def setup_ollama_flow(mock_ollama):  # noqa: F811
    """
    Factory fixture to configure mock_ollama for a standard message + background task flow.

    Sets up a multi-phase handler:
    1. First call: tool call (search) with given query
    2. Second call: message response
    3. Third call onwards: background task response

    Usage:
        setup_ollama_flow(
            search_query="weather forecast",
            message_response="here's the weather! 🌤️",
            background_response="background task response (optional)",
        )
    """

    def _setup(
        search_query: str,
        message_response: str,
        background_response: str = "",
    ) -> None:
        request_count = [0]

        def multi_phase_handler(request: dict, count: int) -> dict:
            request_count[0] += 1
            if request_count[0] == 1:
                # First call: message agent tool call
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": search_query}
                )
            elif request_count[0] == 2:
                # Second call: message agent final response
                return mock_ollama._make_text_response(request, message_response)
            else:
                # Third call onwards: background task
                return mock_ollama._make_text_response(request, background_response)

        mock_ollama.set_response_handler(multi_phase_handler)

    return _setup
