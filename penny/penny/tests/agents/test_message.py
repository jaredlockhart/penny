"""Integration tests for the MessageAgent."""

import pytest
from sqlmodel import select

from penny.database.models import MessageLog, SearchLog
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_response",
    [
        "<function=search><parameter=query>Canadian wildfires</parameter></function>",
        '<tools><search>{"query": "unusual instruments"}</search></tools>',
    ],
    ids=["function-param-xml", "tools-xml"],
)
async def test_xml_tool_call_not_leaked_to_user(
    malformed_response,
    signal_server,
    mock_ollama,
    test_config,
    _mock_search,
    test_user_info,
    running_penny,
):
    """
    Regression test for #262: malformed tool call leaked to user.

    When a model emits XML-like markup in the content field instead of using
    structured tool_calls, the agent retries without consuming an agentic loop
    step, and the clean response reaches the user.
    """
    clean_response = "here are some great movies for you!"

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_text_response(request, malformed_response)
        return mock_ollama._make_text_response(request, clean_response)

    mock_ollama.set_response_handler(handler)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="recommend a movie",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert mock_ollama._request_count >= 2, (
            "Agent should have retried when XML markup was in content"
        )
        assert response["message"] == clean_response


@pytest.mark.asyncio
async def test_basic_message_flow(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """
    Test the complete message flow:
    1. User sends a message via Signal
    2. Penny receives and processes it
    3. Ollama returns a tool call (search)
    4. Search tool executes (mocked)
    5. Ollama returns final response
    6. Penny sends reply via Signal
    """
    # Configure Ollama to return search tool call, then final response
    mock_ollama.set_default_flow(
        search_query="test search query",
        final_response="here's what i found about your question! 🌟",
    )

    async with running_penny(test_config) as penny:
        # Verify we have a WebSocket connection
        assert len(signal_server._websockets) == 1, "Penny should have connected to WebSocket"

        # Send incoming message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what's the weather like today?",
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify the response
        assert response["recipients"] == [TEST_SENDER]
        assert "here's what i found" in response["message"].lower()

        # Verify Ollama was called twice (tool call + final response)
        assert len(mock_ollama.requests) == 2, "Expected 2 Ollama calls (tool + final)"

        # First request should have user message
        first_request = mock_ollama.requests[0]
        messages = first_request.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert any("weather" in m.get("content", "").lower() for m in user_messages)

        # Second request should include tool result
        second_request = mock_ollama.requests[1]
        messages = second_request.get("messages", [])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) >= 1, "Second request should include tool result"

        # Verify typing indicators were sent
        assert len(signal_server.typing_events) >= 1, "Should have sent typing indicator"

        # Verify messages were logged to database
        incoming_messages = penny.db.get_user_messages(TEST_SENDER)
        assert len(incoming_messages) >= 1, "Incoming message should be logged"

        with penny.db.get_session() as session:
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
        assert len(outgoing) >= 1, "Outgoing message should be logged"


@pytest.mark.asyncio
async def test_message_without_tool_call(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """Test handling a message where Ollama doesn't call a tool."""

    # Configure Ollama to return direct response (no tool call)
    def direct_response(request, count):
        return mock_ollama._make_text_response(request, "just a simple response! 🌟")

    mock_ollama.set_response_handler(direct_response)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="hello penny",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert response["recipients"] == [TEST_SENDER]
        assert "simple response" in response["message"].lower()

        # Only one Ollama call (no tool)
        assert len(mock_ollama.requests) == 1


@pytest.mark.asyncio
async def test_profile_context_excludes_dob_and_redacts_name_from_search(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """
    Test privacy protections for user profile data:
    1. DOB is not included in the profile context sent to Ollama
    2. User name is redacted from search queries before reaching Perplexity
    """
    # The test user is "Test User" from conftest — have the model generate
    # a search query that includes the user's name
    mock_ollama.set_default_flow(
        search_query="Test User Toronto weather forecast",
        final_response="here's the weather! 🌤️",
    )

    async with running_penny(test_config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what's the weather?",
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Verify DOB is NOT in the Ollama prompt messages
        first_request = mock_ollama.requests[0]
        messages = first_request.get("messages", [])
        system_messages = [m for m in messages if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_messages)
        assert "1990-01-01" not in all_system_text, "DOB should not be in profile context"
        assert "born" not in all_system_text.lower(), "DOB field should not be in profile context"

        # Verify profile context IS present (name + location)
        assert "Test User" in all_system_text, "Name should be in profile context"
        assert "Seattle" in all_system_text, "Location should be in profile context"

        # Verify user name was redacted from the search query logged to DB
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
        assert len(search_logs) >= 1, "Search should have been logged"
        logged_query = search_logs[0].query
        assert "Test User" not in logged_query, "User name should be redacted from search query"
        assert "Toronto weather forecast" in logged_query, "Rest of query should be preserved"


@pytest.mark.asyncio
async def test_name_not_redacted_when_user_says_it(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """
    When the user's message contains their own name (e.g. searching for
    a celebrity with the same name), the name should NOT be redacted from
    the search query.
    """
    # Model echoes the name back in the search query
    mock_ollama.set_default_flow(
        search_query="Test User celebrity gossip",
        final_response="here's what i found! 🌟",
    )

    async with running_penny(test_config) as penny:
        # User explicitly typed their own name in the message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="search for Test User celebrity gossip",
        )
        await signal_server.wait_for_message(timeout=10.0)

        # Name should be preserved in the search query since the user said it
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
        assert len(search_logs) >= 1, "Search should have been logged"
        logged_query = search_logs[0].query
        assert "Test User" in logged_query, "Name should NOT be redacted when user said it"
