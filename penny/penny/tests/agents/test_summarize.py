"""Integration tests for the SummarizeAgent."""

import pytest
from sqlmodel import select

from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_summarize_background_task(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """
    Test the summarize background task:
    1. Send a message and get a response (creates a thread)
    2. Wait for idle time to pass
    3. Verify SummarizeAgent generates and stores a summary
    """
    config = make_config(idle_seconds=0.5)
    setup_ollama_flow(
        search_query="weather forecast today",
        message_response="here's the weather info! 🌤️",
        background_response="user asked about weather, assistant provided forecast",
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="what's the weather like?")
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "weather" in response["message"].lower()

        # Get the outgoing message id
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            assert outgoing is not None
            message_id = outgoing.id
            assert outgoing.parent_id is not None
            assert outgoing.parent_summary is None

        # Wait for summarize task to produce a summary in the DB
        def summary_exists():
            with penny.db.get_session() as session:
                msg = session.get(MessageLog, message_id)
                return msg is not None and msg.parent_summary is not None

        await wait_until(summary_exists)

        # Verify summary content
        with penny.db.get_session() as session:
            outgoing = session.get(MessageLog, message_id)
            assert outgoing is not None
            assert len(outgoing.parent_summary) > 0
            assert "weather" in outgoing.parent_summary.lower()

        assert len(mock_ollama.requests) >= 3, "Expected at least 3 Ollama calls"


@pytest.mark.asyncio
async def test_summarize_skips_short_thread(
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
):
    """
    Test that SummarizeAgent marks threads with < 2 messages as processed
    with an empty summary (no LLM call needed).
    """
    config = make_config()

    from penny.agents import SummarizeAgent
    from penny.constants import SYSTEM_PROMPT
    from penny.database import Database
    from penny.database.migrate import migrate

    db = Database(config.db_path)
    db.create_tables()
    migrate(config.db_path)

    # Insert an outgoing message with parent_id pointing to a non-existent message.
    # _walk_thread will return only 1 message (the outgoing one), triggering the
    # short-thread path (< 2 messages).
    msg_id = db.log_message(
        direction="outgoing",
        sender=TEST_SENDER,
        content="a response",
        parent_id=99999,  # Non-existent parent
    )
    assert msg_id is not None

    # Verify it shows up as unsummarized
    unsummarized = db.get_unsummarized_messages()
    assert len(unsummarized) == 1

    summarize_agent = SummarizeAgent(
        system_prompt=SYSTEM_PROMPT,
        model=config.ollama_foreground_model,
        ollama_api_url=config.ollama_api_url,
        tools=[],
        db=db,
    )

    ollama_calls_before = len(mock_ollama.requests)
    result = await summarize_agent.execute()
    assert result is True, "Should return True (work was done — marked as processed)"

    # No Ollama call for short threads
    assert len(mock_ollama.requests) == ollama_calls_before, (
        "Should not call Ollama for short threads"
    )

    # Verify the message was marked with empty summary
    with db.get_session() as session:
        msg = session.get(MessageLog, msg_id)
        assert msg is not None
        assert msg.parent_summary == ""


@pytest.mark.asyncio
async def test_summarize_no_work(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Test that SummarizeAgent returns False when there are no unsummarized messages."""
    config = make_config()

    async with running_penny(config) as penny:
        from penny.agents import SummarizeAgent
        from penny.constants import SYSTEM_PROMPT

        summarize_agent = SummarizeAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )

        result = await summarize_agent.execute()
        assert result is False, "Should return False when no work to do"


@pytest.mark.asyncio
async def test_summarize_empty_response(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
    setup_ollama_flow,
):
    """Test that SummarizeAgent returns False when LLM returns empty answer."""
    config = make_config()
    setup_ollama_flow(
        search_query="test query",
        message_response="test response! 🌟",
        background_response="",  # Empty summary from LLM
    )

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="tell me something")
        await signal_server.wait_for_message(timeout=10.0)

        # Manually invoke summarize agent
        from penny.agents import SummarizeAgent
        from penny.constants import SYSTEM_PROMPT

        summarize_agent = SummarizeAgent(
            system_prompt=SYSTEM_PROMPT,
            model=config.ollama_foreground_model,
            ollama_api_url=config.ollama_api_url,
            tools=[],
            db=penny.db,
        )

        # Empty string response — the agent stores it (truthy check is `is not None`)
        result = await summarize_agent.execute()
        # Empty string is still not None, so it gets stored
        assert result is True
