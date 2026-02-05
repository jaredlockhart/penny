"""Integration tests for the basic message flow."""

import asyncio
import contextlib

import pytest

from penny.penny import Penny


@pytest.mark.asyncio
async def test_basic_message_flow(signal_server, ollama_server, test_config, mock_search):
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
    ollama_server.set_default_flow(
        search_query="test search query",
        final_response="here's what i found about your question! 🌟",
    )

    # Create Penny instance
    penny = Penny(test_config)

    # Run Penny in background
    penny_task = asyncio.create_task(penny.run())

    try:
        # Wait for WebSocket connection to be established
        await asyncio.sleep(0.3)

        # Verify we have a WebSocket connection
        assert len(signal_server._websockets) == 1, "Penny should have connected to WebSocket"

        # Send incoming message
        await signal_server.push_message(
            sender="+15559876543",
            content="what's the weather like today?",
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify the response
        assert response["recipients"] == ["+15559876543"]
        assert "here's what i found" in response["message"].lower()

        # Verify Ollama was called twice (tool call + final response)
        assert len(ollama_server.requests) == 2, "Expected 2 Ollama calls (tool + final)"

        # First request should have user message
        first_request = ollama_server.requests[0]
        messages = first_request.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert any("weather" in m.get("content", "").lower() for m in user_messages)

        # Second request should include tool result
        second_request = ollama_server.requests[1]
        messages = second_request.get("messages", [])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) >= 1, "Second request should include tool result"

        # Verify typing indicators were sent
        assert len(signal_server.typing_events) >= 1, "Should have sent typing indicator"

        # Verify messages were logged to database
        incoming_messages = penny.db.get_user_messages("+15559876543")
        assert len(incoming_messages) >= 1, "Incoming message should be logged"

        with penny.db.get_session() as session:
            from sqlmodel import select

            from penny.database.models import MessageLog

            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
        assert len(outgoing) >= 1, "Outgoing message should be logged"

    finally:
        # Clean shutdown
        penny_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await penny_task
        await penny.shutdown()


@pytest.mark.asyncio
async def test_message_without_tool_call(signal_server, ollama_server, test_config, mock_search):
    """Test handling a message where Ollama doesn't call a tool."""

    # Configure Ollama to return direct response (no tool call)
    def direct_response(request, count):
        return ollama_server._make_text_response(request, "just a simple response! 🌟")

    ollama_server.set_response_handler(direct_response)

    penny = Penny(test_config)
    penny_task = asyncio.create_task(penny.run())

    try:
        await asyncio.sleep(0.3)

        await signal_server.push_message(
            sender="+15559876543",
            content="hello penny",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert response["recipients"] == ["+15559876543"]
        assert "simple response" in response["message"].lower()

        # Only one Ollama call (no tool)
        assert len(ollama_server.requests) == 1

    finally:
        penny_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await penny_task
        await penny.shutdown()
