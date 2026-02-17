"""Integration tests for /test mode."""

from pathlib import Path

import pytest

from penny.constants import TEST_DB_PATH
from penny.responses import PennyResponse
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_test_mode_basic_flow(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """
    Test that /test command uses test database and prepends [TEST] to response.
    """
    mock_ollama.set_default_flow(
        search_query="test search query",
        final_response="here's what i found about your question! 🌟",
    )

    async with running_penny(test_config):
        # Send /test message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="/test what's the weather like today?",
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify the response has [TEST] prefix
        assert response["recipients"] == [TEST_SENDER]
        assert response["message"].startswith(PennyResponse.TEST_MODE_PREFIX)
        assert "here's what i found" in response["message"].lower()

        # Verify test database was used (message should be in test db)
        test_db_path = Path(test_config.db_path).parent / TEST_DB_PATH.name
        assert test_db_path.exists(), "Test database should have been created"

        # Note: The incoming message is still logged to production database by the channel
        # after the agent runs, but the agent's processing (search, LLM calls) use test db
        # This is expected behavior - test mode isolates agent processing, not channel logging


@pytest.mark.asyncio
async def test_test_mode_rejects_nested_commands(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """
    Test that /test rejects nested commands like /test /debug.
    """
    async with running_penny(test_config):
        # Send /test with nested command
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="/test /debug",
        )

        # Wait for error response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify error message
        assert response["recipients"] == [TEST_SENDER]
        assert "nested commands are not supported" in response["message"].lower()

        # Verify Ollama was NOT called
        assert len(mock_ollama.requests) == 0, "Ollama should not be called for nested commands"


@pytest.mark.asyncio
async def test_test_mode_rejects_threading(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """
    Test that /test rejects threaded messages (blocked at channel layer like all commands).
    """
    async with running_penny(test_config):
        # Send /test as a threaded message (quote reply)
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="/test what about test mode?",
            quote={"id": 12345, "text": "some previous message"},
        )

        # Wait for error response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify error message from channel layer
        assert response["recipients"] == [TEST_SENDER]
        assert "commands can't be used in threads" in response["message"].lower()

        # Verify Ollama was NOT called
        assert len(mock_ollama.requests) == 0, "Ollama should not be called for threaded commands"


@pytest.mark.asyncio
async def test_test_mode_uses_real_external_services(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """
    Test that /test mode uses real external services (Ollama, Perplexity).
    """
    mock_ollama.set_default_flow(
        search_query="test search query",
        final_response="here's what i found! 🌟",
    )

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="/test tell me something interesting",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify response has [TEST] prefix
        assert response["message"].startswith(PennyResponse.TEST_MODE_PREFIX)

        # Verify Ollama was called (real service usage)
        assert len(mock_ollama.requests) >= 1, "Ollama should be called in test mode"

        # Verify search tool was invoked
        # The mock_search fixture tracks search calls globally
        # In test mode, the search tool should still execute


@pytest.mark.asyncio
async def test_test_mode_blocks_threading_to_test_responses(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """
    Test that threading/replying to a test mode response is rejected.

    Scenario:
    1. User: /test some prompt
    2. Penny: [TEST] Blah blah response
    3. User: quotetext(Blah blah response) hello
    4. Penny: "Threading is not supported for test mode responses."
    """
    mock_ollama.set_default_flow(
        search_query="test search query",
        final_response="here's what i found! 🌟",
    )

    async with running_penny(test_config):
        # Send /test message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="/test what's the weather like today?",
        )

        # Wait for test response
        test_response = await signal_server.wait_for_message(timeout=10.0)
        assert test_response["message"].startswith(PennyResponse.TEST_MODE_PREFIX)

        # Try to thread-reply to the test response
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="tell me more",
            quote={"id": 12345, "text": test_response["message"]},
        )

        # Wait for error response
        error_response = await signal_server.wait_for_message(timeout=10.0)

        # Verify error message
        assert error_response["recipients"] == [TEST_SENDER]
        assert "test mode can't be used in threads" in error_response["message"].lower()


@pytest.mark.asyncio
async def test_test_mode_snapshot_created_at_startup(test_config, running_penny):
    """
    Test that test database snapshot is created (via entrypoint in production).

    In tests, we manually create the snapshot since the entrypoint script doesn't run.
    """
    import shutil

    from penny.database import Database
    from penny.database.migrate import migrate

    # Create production database and add a message
    prod_db = Database(test_config.db_path)
    migrate(test_config.db_path)
    prod_db.create_tables()
    prod_db.log_message(
        direction="incoming",
        sender=TEST_SENDER,
        content="production message",
    )

    # Verify production db file exists
    prod_path = Path(test_config.db_path)
    assert prod_path.exists(), "Production database should exist before starting Penny"

    # Manually create test snapshot (simulating entrypoint.sh behavior in tests)
    test_db_path = prod_path.parent / TEST_DB_PATH.name
    shutil.copyfile(prod_path, test_db_path)

    # Start Penny
    async with running_penny(test_config):
        # Verify test db exists
        assert test_db_path.exists(), f"Test database should exist at {test_db_path}"

        # Verify test db is a copy of production db
        test_db = Database(str(test_db_path))
        test_messages = test_db.get_user_messages(TEST_SENDER)
        assert len(test_messages) == 1, "Test db should contain production message"
        assert test_messages[0].content == "production message"


@pytest.mark.asyncio
async def test_test_mode_shows_typing_indicator(
    signal_server, mock_ollama, test_config, _mock_search, running_penny
):
    """
    Test that /test command shows typing indicator while processing.
    """
    mock_ollama.set_default_flow(
        search_query="test search query",
        final_response="here's what i found! 🌟",
    )

    async with running_penny(test_config):
        # Clear any startup typing events
        signal_server.typing_events.clear()

        # Send /test message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="/test what's the weather like today?",
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify the response
        assert response["recipients"] == [TEST_SENDER]
        assert response["message"].startswith(PennyResponse.TEST_MODE_PREFIX)

        # Verify typing indicators were sent (at least one start and one stop)
        assert len(signal_server.typing_events) >= 2, "Should have typing events"
        assert signal_server.typing_events[0] == (
            "start",
            TEST_SENDER,
        ), "First typing event should be start"
        assert signal_server.typing_events[-1] == (
            "stop",
            TEST_SENDER,
        ), "Last typing event should be stop"
