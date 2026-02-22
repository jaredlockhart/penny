"""Integration tests for Signal channel."""

import pytest

from penny.channels.signal import SignalChannel
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_validate_connectivity_success(signal_server, test_config, mock_ollama):
    """Test that validate_connectivity succeeds with a reachable Signal API."""
    from penny.agents import MessageAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=test_config.ollama_api_url,
        model=test_config.ollama_foreground_model,
        db=db,
        max_retries=test_config.ollama_max_retries,
        retry_delay=test_config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=test_config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    # Should not raise
    await channel.validate_connectivity()

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_dns_failure(test_db, mock_ollama):
    """Test that validate_connectivity raises ConnectionError on DNS failure."""
    from penny.agents import MessageAgent
    from penny.config import Config
    from penny.prompts import Prompt

    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://nonexistent-hostname-that-will-never-resolve.invalid:8080",
        discord_bot_token=None,
        discord_channel_id=None,
        ollama_api_url="http://localhost:11434",
        ollama_foreground_model="test-model",
        ollama_background_model="test-model",
        perplexity_api_key=None,
        log_level="DEBUG",
        db_path=test_db,
    )

    db = Database(config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=config.ollama_api_url,
        model=config.ollama_foreground_model,
        db=db,
        max_retries=config.ollama_max_retries,
        retry_delay=config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await channel.validate_connectivity()

    error_message = str(exc_info.value)
    assert "Cannot resolve Signal API hostname" in error_message
    assert "nonexistent-hostname-that-will-never-resolve.invalid" in error_message
    assert "SIGNAL_API_URL" in error_message

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_connection_refused(test_db, mock_ollama):
    """Test that validate_connectivity raises ConnectionError when server is unreachable."""
    from penny.agents import MessageAgent
    from penny.config import Config
    from penny.prompts import Prompt

    # Use localhost on a port that's not listening
    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://localhost:19999",  # Unlikely to be in use
        discord_bot_token=None,
        discord_channel_id=None,
        ollama_api_url="http://localhost:11434",
        ollama_foreground_model="test-model",
        ollama_background_model="test-model",
        perplexity_api_key=None,
        log_level="DEBUG",
        db_path=test_db,
    )

    db = Database(config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=config.ollama_api_url,
        model=config.ollama_foreground_model,
        db=db,
        max_retries=config.ollama_max_retries,
        retry_delay=config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await channel.validate_connectivity()

    error_message = str(exc_info.value)
    assert "Cannot connect to Signal API" in error_message
    assert "http://localhost:19999" in error_message

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_rejects_empty_without_attachments(
    signal_server, test_config, mock_ollama
):
    """Test that send_message raises ValueError for empty text with no attachments."""
    from penny.agents import MessageAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=test_config.ollama_api_url,
        model=test_config.ollama_foreground_model,
        db=db,
        max_retries=test_config.ollama_max_retries,
        retry_delay=test_config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=test_config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ValueError, match="Cannot send empty"):
        await channel.send_message(TEST_SENDER, "", attachments=None, quote_message=None)

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_allows_empty_text_with_attachments(
    signal_server, test_config, mock_ollama
):
    """Test that send_message succeeds with empty text when attachments are provided."""
    from penny.agents import MessageAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=test_config.ollama_api_url,
        model=test_config.ollama_foreground_model,
        db=db,
        max_retries=test_config.ollama_max_retries,
        retry_delay=test_config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=test_config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    # Should not raise — empty text is fine when attachments are present
    fake_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    result = await channel.send_message(
        TEST_SENDER, "", attachments=[fake_image], quote_message=None
    )
    assert result is not None

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_retries_on_socket_exception_400(
    signal_server, test_config, mock_ollama
):
    """Test that send_message retries when signal-cli returns a 400 SocketException."""
    from penny.agents import MessageAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=test_config.ollama_api_url,
        model=test_config.ollama_foreground_model,
        db=db,
        max_retries=test_config.ollama_max_retries,
        retry_delay=test_config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=test_config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
        max_retries=2,
        retry_delay=0.01,
    )

    # Queue one transient SocketException 400 then let the retry succeed
    socket_error_body = {
        "error": (
            "Failed to send message: Failed to get response for request"
            " (SocketException) (UnexpectedErrorException)"
        )
    }
    signal_server.queue_send_error(400, socket_error_body)

    result = await channel.send_message(TEST_SENDER, "hello", attachments=None, quote_message=None)

    # Should have succeeded on the retry
    assert result is not None
    # The message was eventually delivered (the successful send is captured)
    assert len(signal_server.outgoing_messages) == 1

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_no_retry_on_non_transient_400(signal_server, test_config, mock_ollama):
    """Test that send_message does NOT retry on non-transient 400 errors."""
    from penny.agents import MessageAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=test_config.ollama_api_url,
        model=test_config.ollama_foreground_model,
        db=db,
        max_retries=test_config.ollama_max_retries,
        retry_delay=test_config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=test_config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
        max_retries=2,
        retry_delay=0.01,
    )

    # Queue a non-transient 400 (bad recipient format — should not retry)
    signal_server.queue_send_error(400, {"error": "Invalid recipient number"})
    # Queue another 200 success that should NOT be reached if retry is skipped
    signal_server.queue_send_error(400, {"error": "Invalid recipient number"})

    result = await channel.send_message(TEST_SENDER, "hello", attachments=None, quote_message=None)

    # Should have returned None without retrying
    assert result is None
    # No messages should have been captured by the success handler
    assert len(signal_server.outgoing_messages) == 0
    # The second queued error should still be in the queue (retry was not attempted)
    assert len(signal_server._send_response_queue) == 1

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_gives_up_after_max_retries(signal_server, test_config, mock_ollama):
    """Test that send_message returns None after exhausting retries on persistent errors."""
    from penny.agents import MessageAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

    client = OllamaClient(
        api_url=test_config.ollama_api_url,
        model=test_config.ollama_foreground_model,
        db=db,
        max_retries=test_config.ollama_max_retries,
        retry_delay=test_config.ollama_retry_delay,
    )
    message_agent = MessageAgent(
        system_prompt=Prompt.SEARCH_PROMPT,
        background_model_client=client,
        foreground_model_client=client,
        tools=[],
        db=db,
        config=test_config,
        max_steps=1,
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
        max_retries=2,
        retry_delay=0.01,
    )

    # Queue 3 transient errors (initial attempt + 2 retries = 3 total)
    socket_error = {"error": "Failed to send message: (SocketException)"}
    for _ in range(3):
        signal_server.queue_send_error(400, socket_error)

    result = await channel.send_message(TEST_SENDER, "hello", attachments=None, quote_message=None)

    # All retries exhausted — should return None
    assert result is None
    # No successful sends
    assert len(signal_server.outgoing_messages) == 0
    # All 3 queued errors were consumed
    assert len(signal_server._send_response_queue) == 0

    await channel.close()
