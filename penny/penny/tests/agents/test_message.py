"""Integration tests for ChatAgent message handling.

Test organization:
1. Full integration (happy path) — comprehensive end-to-end message flow
2. Special success cases — no tool call, privacy redaction, anti-refusal
3. Error / edge cases — XML leak regression, short response warning, delivery failure
"""

from datetime import datetime

import pytest
from sqlmodel import select

from penny.constants import PennyConstants
from penny.database.models import MessageLog, SearchLog
from penny.tests.conftest import TEST_SENDER, wait_until

# ── 1. Full integration (happy path) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_message_flow(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    mock_serper_image,
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
    config = make_config(serper_api_key="test-key")

    # Configure Ollama to return search tool call, then final response
    mock_ollama.set_default_flow(
        search_query="test search query",
        final_response="here's what i found about your question! 🌟",
    )

    async with running_penny(config) as penny:
        # Verify we have a WebSocket connection
        assert len(signal_server._websockets) == 1, "Penny should have connected to WebSocket"

        # Seed full context: weekly history, daily history, notified thought, dislike
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=datetime(2026, 3, 16),
            period_end=datetime(2026, 3, 23),
            duration=PennyConstants.HistoryDuration.WEEKLY,
            topics="- Guitar pedal research\n- Quantum computing news",
        )
        penny.db.history.add(
            user=TEST_SENDER,
            period_start=datetime(2026, 3, 23),
            period_end=datetime(2026, 3, 24),
            duration=PennyConstants.HistoryDuration.DAILY,
            topics="- Tone King Royalist amp",
        )
        thought = penny.db.thoughts.add(TEST_SENDER, "Recent thought about amps")
        if thought:
            penny.db.thoughts.mark_notified(thought.id)
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Country music",
            valence="negative",
        )

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

        # Full system prompt structure assertion
        system_text = [
            m.get("content", "") for m in first_request["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

## Identity
You are Penny. You and the user are friends who text regularly. \
This is mid-conversation — not a fresh chat.

Voice:
- Reply like you're continuing a text thread.
- React to what the user actually said before giving information. \
If they corrected you, own it. If they expressed excitement, match it. \
If they asked a follow-up, connect it to what came before.
- Present information naturally but you can still use short formatted blocks \
(bold names, links) when listing products or facts. \
Just wrap them in conversational text, not a clinical dump.
- Finish every message with an emoji.

## Context
### User Profile
The user's name is Test User.

### Conversation History
Week of Mar 16:
- Guitar pedal research
- Quantum computing news
Mar 23:
- Tone King Royalist amp

### Recent Background Thinking
Recent thought about amps

## Instructions
The user is talking to you — no greetings, no sign-offs, just pick up \
the thread. You have context injected above — \
recent conversation history, relevant knowledge, recent events, \
and your own recent thoughts.

You have tools available:
- **fetch**: Look things up. Pass up to 5 queries and/or URLs.

Every tool call has a `reasoning` field — use it to think out loud. \
Explain what you're looking for, what you already know, \
and what you'll do with the result.

Use your tools to look up information before replying when the user asks \
about something you could look up. \
The only exception is pure greetings with zero topic content ('hey', 'hi') \
or follow-ups where you already have the information from a previous tool call.

When a 'Current Browser Page' section appears above, the user is browsing \
that page right now. If they say 'this page', 'this thread', 'this article', \
or anything ambiguous, they mean the Current Browser Page — not something \
from earlier in the conversation.

Go WIDE: cover as many angles of the user's question as possible. \
Pack up to 5 lookups into a single tool call to explore \
different facets, then do follow-up calls to fill gaps. The user wants a \
comprehensive picture, not a narrow slice — they can drill in afterward.

Every fact, name, and detail in your response must come from your tool \
results or injected context.

Search results contain a 'Sources:' section at the bottom with real URLs. \
When you reference something from a search, use ONLY these source URLs. \
Copy them exactly — character for character. If a topic has no matching \
source URL, mention it without a URL.

When the user changes topics, just go with it. \
If your tools return few results, say what you found and offer to dig deeper.

Always include specific details (specs, dates, prices) and at least one \
source URL so the user can follow up."""
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"

        # Verify typing indicators were sent
        assert len(signal_server.typing_events) >= 1, "Should have sent typing indicator"

        # Verify messages were logged to database
        incoming_messages = penny.db.messages.get_user_messages(TEST_SENDER)
        assert len(incoming_messages) >= 1, "Incoming message should be logged"

        with penny.db.get_session() as session:
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
        assert len(outgoing) >= 1, "Outgoing message should be logged"

        # Verify device_id FK is populated on both incoming and outgoing
        test_device = penny.db.devices.get_by_identifier(TEST_SENDER)
        assert test_device is not None, "Test device should be registered"
        assert incoming_messages[0].device_id == test_device.id
        assert outgoing[0].device_id == test_device.id

        # Verify search logs have default trigger
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
        if search_logs:
            assert search_logs[0].trigger == "user_message"

        # No conversation echo thoughts should be logged
        # (old _log_conversation_thought is removed; thoughts come from tool reasoning only)
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        conversation_echoes = [
            t for t in thoughts if t.content.startswith("Conversation: user said")
        ]
        assert len(conversation_echoes) == 0, "Conversation echo thoughts should not be logged"

        # Serper image search should use the model's search query, not full content
        mock_serper_image.assert_called_once()
        image_query = mock_serper_image.call_args[0][0]
        assert image_query == "test search query"
        assert len(image_query) <= 300

        # Outgoing message should have an image attachment
        assert response.get("base64_attachments"), "Response should include an image attachment"


# ── 2. Special success cases ──────────────────────────────────────────────


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
        all_text = " ".join(m.get("content", "") for m in messages)
        assert "1990-01-01" not in all_text, "DOB should not be in profile context"
        assert "born" not in all_text.lower(), "DOB field should not be in profile context"

        # Verify profile context IS present (name only, not location)
        assert "Test User" in all_text, "Name should be in profile context"
        assert "Seattle" not in all_text, "Location should not be in profile context"

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


@pytest.mark.asyncio
async def test_conversation_prompt_includes_antirefusal_instruction(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """
    Regression test for #775: CONVERSATION_PROMPT must include an explicit instruction
    to never refuse a request, so the model always provides something useful.
    """
    mock_ollama.set_default_flow(
        search_query="vegan restaurants downtown",
        final_response="here are some vegan options! 🌱",
    )

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what are the best vegan restaurants?",
        )
        await signal_server.wait_for_message(timeout=10.0)

    # Verify the system prompt instructs the model to always provide something useful
    first_request = mock_ollama.requests[0]
    messages = first_request.get("messages", [])
    system_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
    assert "offer to dig deeper" in system_text.lower(), (
        "CONVERSATION_PROMPT should instruct the model to always offer help"
    )


# ── 3. Error / edge cases ────────────────────────────────────────────────


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
async def test_short_response_logged_as_warning(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny, caplog
):
    """
    Regression test for #775: short/apologetic responses should be logged as warnings.

    When the model returns a very short response (< 10 words), a warning is logged
    to make these cases visible for debugging. The response is still delivered to
    the user — the warning is diagnostic, not suppressive.
    """
    import logging

    apologetic_response = "I'm sorry, but I can't help with that."

    def handler(request, count):
        return mock_ollama._make_text_response(request, apologetic_response)

    mock_ollama.set_response_handler(handler)

    with caplog.at_level(logging.WARNING, logger="penny.agents.base"):
        async with running_penny(test_config):
            await signal_server.push_message(
                sender=TEST_SENDER,
                content="what are the best vegan restaurants in downtown metropolis?",
            )
            response = await signal_server.wait_for_message(timeout=10.0)

    # Response is still delivered to the user
    assert response["message"] == apologetic_response

    # Warning was logged for the short response
    short_response_warnings = [r for r in caplog.records if "Short response detected" in r.message]
    assert len(short_response_warnings) >= 1, "Should log a warning for short responses"


@pytest.mark.asyncio
async def test_delivery_failure_sends_notice(
    signal_server, mock_ollama, test_config, _mock_search, test_user_info, running_penny
):
    """Test that a delivery failure notice is sent to the user when all send retries fail.

    When signal-cli returns a 400 SocketException on every attempt, the channel
    exhausts its retries and returns None from send_message.  _dispatch_to_agent
    should detect this and send a brief failure notice so the user knows to retry.
    """
    mock_ollama.set_default_flow(
        search_query="test query",
        final_response="my answer to your question",
    )

    # test_config uses ollama_max_retries=1, so SignalChannel makes 2 total send
    # attempts (attempt 0 + 1 retry) for the main response.  Queue 2 transient
    # SocketException errors to exhaust those attempts; the 3rd request (the
    # failure notice) gets the default 200 success.
    socket_error = {
        "error": (
            "Failed to send message: Failed to get response for request"
            " (SocketException) (UnexpectedErrorException)"
        )
    }
    signal_server.queue_send_error(400, socket_error)
    signal_server.queue_send_error(400, socket_error)

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="hello there")

        await wait_until(lambda: len(signal_server.outgoing_messages) >= 1)

        notice = signal_server.outgoing_messages[0]
        assert notice["recipients"] == [TEST_SENDER]
        assert "trouble" in notice["message"].lower()
        assert len(mock_ollama.requests) == 2
