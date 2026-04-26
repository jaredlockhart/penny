"""Integration tests for ChatAgent message handling.

Test organization:
1. Full integration (happy path) — comprehensive end-to-end message flow
2. Special success cases — no tool call, anti-refusal
3. Error / edge cases — XML leak regression, short response warning, delivery failure
"""

from unittest.mock import AsyncMock

import pytest
from sqlmodel import select

from penny.database.memory_store import EntryInput, LogEntryInput, RecallMode
from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, wait_until

# ── 1. Full integration (happy path) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_message_flow(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test the complete message flow:
    1. User sends a message via Signal
    2. Penny receives and processes it
    3. Ollama returns a tool call (fetch)
    4. Fetch tool executes (mocked)
    5. Ollama returns final response
    6. Penny sends reply via Signal
    """
    config = make_config()

    # Configure Ollama to return fetch tool call, then final response
    mock_llm.set_default_flow(
        final_response="here's what i found about your question! 🌟",
    )

    async with running_penny(config) as penny:
        # Verify we have a WebSocket connection
        assert len(signal_server._websockets) == 1, "Penny should have connected to WebSocket"

        # Seed full context: notified thought, dislike, active memory
        thought = penny.db.thoughts.add(TEST_SENDER, "Recent thought about amps")
        if thought:
            penny.db.thoughts.mark_notified(thought.id)
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Country music",
            valence="negative",
        )
        # Memory seed: exercise every rendering path in one verbatim assertion.
        # Alphabetical by name — "likes" < "old-facts" < "secrets" < "tips".
        penny.db.memories.create_collection("likes", "positive prefs", RecallMode.ALL)
        penny.db.memories.write(
            "likes",
            [EntryInput(key="dark roast", content="loves dark roast")],
            author="user",
        )
        penny.db.memories.create_log("tips", "useful tips", RecallMode.RECENT)
        penny.db.memories.append(
            "tips", [LogEntryInput(content="tune before playing")], author="user"
        )
        # Off and archived memories are seeded with entries so the verbatim
        # prompt assertion below proves they are filtered out of ambient recall.
        penny.db.memories.create_collection("secrets", "hidden", RecallMode.OFF)
        penny.db.memories.write(
            "secrets",
            [EntryInput(key="do-not-share", content="classified")],
            author="user",
        )
        penny.db.memories.create_collection("old-facts", "archived", RecallMode.ALL)
        penny.db.memories.write(
            "old-facts",
            [EntryInput(key="stale", content="no longer relevant")],
            author="user",
        )
        penny.db.memories.archive("old-facts")

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
        assert len(mock_llm.requests) == 2, "Expected 2 Ollama calls (tool + final)"

        # First request should have user message
        first_request = mock_llm.requests[0]
        messages = first_request.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert any("weather" in m.get("content", "").lower() for m in user_messages)

        # Second request should include tool result
        second_request = mock_llm.requests[1]
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

### likes
positive prefs

- [dark roast] loves dark roast

### tips
useful tips

- tune before playing

## Instructions
The user is talking to you — no greetings, no sign-offs, just pick up \
the thread.

You have tools available:
- **browse**: Look things up. Pass up to 3 queries and/or URLs.

Every tool call has a `reasoning` field — use it to think out loud. \
Explain what you're looking for, what you already know, \
and what you'll do with the result.

Use your tools to look up information before replying when the user mentions \
a product, topic, or anything you could look up — even if it appeared in \
Related Past Messages or Knowledge. Past messages tell you what was discussed, \
not the facts about those things. The Knowledge section contains factual \
summaries of pages previously read — use this as background context but always \
verify with fresh lookups when the user asks specific questions. \
The ONLY exception is pure greetings ('hey', 'hi') \
or direct follow-ups to a tool call you made earlier in THIS conversation.

When a 'Current Browser Page' section appears above, the user is browsing \
that page right now. If they say 'this page', 'this thread', 'this article', \
or anything ambiguous, they mean the Current Browser Page — not something \
from earlier in the conversation.

How to use your tools:
1. If the user gave you URLs, read them directly — pass the URLs in the \
queries array. Do NOT search for a site the user already linked.
2. If the user gave you a topic (no URLs), search first to discover \
relevant pages.
3. Read the most promising pages by passing their URLs in the queries \
array (e.g., queries: ["https://example.com/page"]). \
Real pages have full details that search snippets leave out.

After reading pages, you MUST respond with what you found. Do not make \
additional tool calls to re-fetch or supplement pages you already read. \
If a page had limited content, report what was there.

Do NOT answer from search snippets alone — read actual pages first.

Every fact, name, and detail in your response must come from pages you \
read or injected context — not from search snippet summaries.

Search results contain a 'Sources:' section at the bottom with real URLs. \
When you reference something from a search, use ONLY these source URLs. \
Copy them exactly — character for character. If a topic has no matching \
source URL, mention it without a URL.

When the user changes topics, just go with it.

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

        # No conversation echo thoughts should be logged
        # (old _log_conversation_thought is removed; thoughts come from tool reasoning only)
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        conversation_echoes = [
            t for t in thoughts if t.content.startswith("Conversation: user said")
        ]
        assert len(conversation_echoes) == 0, "Conversation echo thoughts should not be logged"


# ── 1b. Ambient-recall integration cases ─────────────────────────────────


@pytest.mark.asyncio
async def test_chat_prompt_renders_relevant_mode_via_embedding(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A collection with recall=relevant surfaces its matching entry in the chat prompt.

    The entry's content_embedding and the current-message embedding are both
    the same unit vector (cosine=1), so the entry ranks first against the
    0.0 floor.  A second orthogonal entry stays below nothing — with floor=0.0
    it's included too, but only the matching one is asserted.
    """
    config = make_config()
    match_vec = [1.0, 0.0, 0.0]

    async with running_penny(config) as penny:
        penny.db.memories.create_collection("knowledge", "facts", RecallMode.RELEVANT)
        penny.db.memories.write(
            "knowledge",
            [
                EntryInput(
                    key="espresso",
                    content="espresso uses 9 bars of pressure",
                    content_embedding=match_vec,
                )
            ],
            author="user",
        )

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=lambda texts: [match_vec] * len(texts))
        penny.chat_agent._embedding_model_client = mock_client
        penny.chat_agent._pending_page_context = None

        prompt = await penny.chat_agent._build_system_prompt(
            TEST_SENDER, content="tell me about espresso"
        )

    assert "### knowledge" in prompt
    assert "facts" in prompt
    assert "- [espresso] espresso uses 9 bars of pressure" in prompt


@pytest.mark.asyncio
async def test_chat_prompt_renders_conversation_pair(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """user-messages + penny-messages logs render as one merged Conversation section.

    Entries are sorted by created_at; author prefixes come from each entry's
    author field.  The secondary member (penny-messages) does not appear under
    its own header — only inside the merged block.
    """
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.memories.create_log("user-messages", "user utterances", RecallMode.RECENT)
        penny.db.memories.create_log("penny-messages", "penny replies", RecallMode.RECENT)
        penny.db.memories.append(
            "user-messages",
            [LogEntryInput(content="how's the weather?")],
            author="user",
        )
        penny.db.memories.append(
            "penny-messages",
            [LogEntryInput(content="clear skies today")],
            author="penny",
        )
        penny.chat_agent._pending_page_context = None

        prompt = await penny.chat_agent._build_system_prompt(TEST_SENDER, content="thanks!")

    assert "### Conversation" in prompt
    assert "[user] how's the weather?" in prompt
    assert "[penny] clear skies today" in prompt
    # Secondary member never gets its own section
    assert "### penny-messages" not in prompt


# ── 2. Special success cases ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_message_without_tool_call(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """Test handling a message where Ollama doesn't call a tool."""

    # Configure Ollama to return direct response (no tool call)
    def direct_response(request, count):
        return mock_llm._make_text_response(request, "just a simple response! 🌟")

    mock_llm.set_response_handler(direct_response)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="hello penny",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert response["recipients"] == [TEST_SENDER]
        assert "simple response" in response["message"].lower()

        # Only one Ollama call (no tool)
        assert len(mock_llm.requests) == 1


@pytest.mark.asyncio
async def test_conversation_prompt_includes_antirefusal_instruction(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """
    Regression test for #775: CONVERSATION_PROMPT must include an explicit instruction
    to never refuse a request, so the model always provides something useful.
    """
    mock_llm.set_default_flow(
        final_response="here are some vegan options! 🌱",
    )

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what are the best vegan restaurants?",
        )
        await signal_server.wait_for_message(timeout=10.0)

    # Verify the system prompt instructs the model to always provide something useful
    first_request = mock_llm.requests[0]
    messages = first_request.get("messages", [])
    system_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
    assert "must respond with what you found" in system_text.lower(), (
        "CONVERSATION_PROMPT should instruct the model to respond with available results"
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
    mock_llm,
    test_config,
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
            return mock_llm._make_text_response(request, malformed_response)
        return mock_llm._make_text_response(request, clean_response)

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="recommend a movie",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert mock_llm._request_count >= 2, (
            "Agent should have retried when XML markup was in content"
        )
        assert response["message"] == clean_response


@pytest.mark.asyncio
async def test_short_response_logged_as_warning(
    signal_server, mock_llm, test_config, test_user_info, running_penny, caplog
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
        return mock_llm._make_text_response(request, apologetic_response)

    mock_llm.set_response_handler(handler)

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
async def test_signal_progress_reactions_track_tool_calls(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """Penny's progress is shown as a morphing emoji reaction on the user's message.

    The dispatch loop reacts to the user's incoming message with 💭 (thinking)
    immediately, swaps to a tool-specific emoji as each tool batch fires
    (🔍 for search, 📖 for read), and removes the reaction once the agent
    finishes. The final response is sent via the normal send path so it
    carries text + image attachments + quote-replies just like before — no
    in-place message editing, no orphan thinking bubble.
    """
    final_answer = "here's the weather forecast for your area! 🌤️"
    # set_default_flow returns a single browse tool call (search query),
    # then the final text response.
    mock_llm.set_default_flow(final_response=final_answer, search_query="weather today")

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="what's the weather?")

        final_msg = await signal_server.wait_for_message_containing(final_answer)

        # The final response is a single fresh message (no edit, no
        # follow-up split). It carries the full agent answer.
        assert final_msg["recipients"] == [TEST_SENDER]
        assert final_answer in final_msg["message"]

        # And only one outgoing message — no "thinking..." bubble, no
        # follow-up image bubble, no edits.
        response_bubbles = [
            m for m in signal_server.outgoing_messages if m.get("message") == final_msg["message"]
        ]
        assert len(response_bubbles) == 1

        # Reactions: 💭 sent at start, 🔍 swapped in once the search tool
        # batch fires, then a remove at delivery time. All three target the
        # same incoming message (the user's question).
        ops = [(e["op"], e.get("reaction")) for e in signal_server.reaction_events]
        assert ("send", "\U0001f4ad") in ops, f"expected initial 💭, got {ops}"
        assert ("send", "\U0001f50d") in ops, f"expected 🔍 search reaction, got {ops}"
        assert ops[-1][0] == "remove", f"final op should be a clear, got {ops}"

        # Every reaction (send and remove) is targeted at the user's
        # incoming message — same target_author + target timestamp throughout.
        targets = {
            (e.get("target_author"), e.get("timestamp")) for e in signal_server.reaction_events
        }
        assert len(targets) == 1, f"reactions should target a single message, got {targets}"


@pytest.mark.asyncio
async def test_signal_progress_reaction_uses_read_emoji_for_url_query(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """When the agent's tool call is a URL fetch (not a text search), the
    progress reaction morphs to 📖 instead of 🔍.

    Covers the URL branch of BrowseTool.to_progress_emoji — the search-only
    test above only exercises the text-query branch.
    """
    final_answer = "great article! 📖"
    # Drive the default flow with a URL query so BrowseTool.to_progress_emoji
    # picks the 📖 (reading) branch instead of 🔍 (searching).
    mock_llm.set_default_flow(
        final_response=final_answer,
        search_query="https://example.com/article",
    )

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="read this for me")
        await signal_server.wait_for_message_containing(final_answer)

        ops = [(e["op"], e.get("reaction")) for e in signal_server.reaction_events]
        assert ("send", "\U0001f4d6") in ops, f"expected 📖 read reaction, got {ops}"
        assert ("send", "\U0001f50d") not in ops, (
            f"URL queries should not trigger the search emoji, got {ops}"
        )


@pytest.mark.asyncio
async def test_signal_progress_clears_reaction_on_failure(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """If the agent crashes mid-run the dispatch loop must still clear the
    reaction so the user isn't left with a stale 💭 on their message forever.
    """

    def boom(request, count):
        raise RuntimeError("simulated agent failure")

    mock_llm.set_response_handler(boom)

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="hello there")

        # Wait for the dispatch loop to set the initial reaction and then
        # clear it from the finally block.
        await wait_until(lambda: any(e["op"] == "remove" for e in signal_server.reaction_events))

        ops = [(e["op"], e.get("reaction")) for e in signal_server.reaction_events]
        assert ops[0] == ("send", "\U0001f4ad"), f"expected initial 💭 send, got {ops}"
        assert any(o[0] == "remove" for o in ops), f"clear must happen on failure, got {ops}"


@pytest.mark.asyncio
async def test_delivery_failure_sends_notice(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """Test that a delivery failure notice is sent to the user when all send retries fail.

    When signal-cli returns a 400 SocketException on every attempt, the channel
    exhausts its retries and returns None from send_message.  _dispatch_to_agent
    should detect this and send a brief failure notice so the user knows to retry.
    """
    mock_llm.set_default_flow(
        final_response="my answer to your question",
    )

    # test_config uses llm_max_retries=1, so SignalChannel makes 2 total send
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

        notice = await signal_server.wait_for_message_containing("trouble")
        assert notice["recipients"] == [TEST_SENDER]
        assert len(mock_llm.requests) == 2
