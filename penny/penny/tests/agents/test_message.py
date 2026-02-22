"""Integration tests for the MessageAgent."""

import pytest
from sqlmodel import select

from penny.database.models import MessageLog, SearchLog
from penny.ollama.embeddings import serialize_embedding
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

        # Verify search logs have default trigger
        with penny.db.get_session() as session:
            search_logs = list(session.exec(select(SearchLog)).all())
        if search_logs:
            assert search_logs[0].trigger == "user_message"


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

        # Verify profile context IS present (name only, not location)
        assert "Test User" in all_system_text, "Name should be in profile context"
        assert "Seattle" not in all_system_text, "Location should not be in profile context"

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
async def test_entity_context_responds_from_knowledge(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """
    When entity knowledge is sufficient to answer the question,
    the agent uses KNOWLEDGE_PROMPT (not SEARCH_PROMPT) and can
    respond directly without searching.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    # Embed handler returns identical vectors → high cosine similarity
    def embed_handler(model, input_text):
        texts = [input_text] if isinstance(input_text, str) else input_text
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

    mock_ollama.set_embed_handler(embed_handler)

    # Direct response (no tool call) since knowledge is sufficient
    def handler(request, count):
        return mock_ollama._make_text_response(
            request, "the KEF LS50 Meta costs $1,599 per pair! 🎵"
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed entity with embedding and enough facts for sufficiency
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "Costs $1,599 per pair")
        penny.db.add_fact(entity.id, "Features Metamaterial Absorption Technology")
        penny.db.add_fact(entity.id, "Made by KEF, a British audio company")
        penny.db.update_entity_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        await signal_server.push_message(
            sender=TEST_SENDER, content="how much does the KEF LS50 cost?"
        )
        response = await signal_server.wait_for_message(timeout=10.0)

        assert "1,599" in response["message"]

        # Verify KNOWLEDGE_PROMPT used (not SEARCH_PROMPT)
        first_request = mock_ollama.requests[0]
        system_msgs = [m for m in first_request["messages"] if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)
        assert "relevant knowledge" in all_system_text.lower()
        assert "You MUST call the search tool" not in all_system_text

        # Entity context was injected
        assert "kef ls50 meta" in all_system_text.lower()
        assert "$1,599" in all_system_text

        # Only 1 Ollama chat call (no search tool call)
        assert len(mock_ollama.requests) == 1


@pytest.mark.asyncio
async def test_entity_context_searches_when_insufficient(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """
    When entity knowledge exists but similarity is below threshold,
    the agent uses SEARCH_PROMPT and forces search.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    # Embed handler: message vector is orthogonal to entity vector → low similarity
    call_count = [0]

    def embed_handler(model, input_text):
        call_count[0] += 1
        texts = [input_text] if isinstance(input_text, str) else input_text
        # Message embedding is orthogonal to entity embedding
        return [[0.0, 1.0, 0.0, 0.0]] * len(texts)

    mock_ollama.set_embed_handler(embed_handler)
    mock_ollama.set_default_flow(
        search_query="weather forecast",
        final_response="here's the weather! 🌤️",
    )

    async with running_penny(config) as penny:
        # Seed entity with embedding (orthogonal to message)
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "Costs $1,599 per pair")
        penny.db.update_entity_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        await signal_server.push_message(sender=TEST_SENDER, content="what's the weather today?")
        await signal_server.wait_for_message(timeout=10.0)

        # SEARCH_PROMPT used (mandatory search)
        first_request = mock_ollama.requests[0]
        system_msgs = [m for m in first_request["messages"] if m.get("role") == "system"]
        all_system_text = " ".join(m.get("content", "") for m in system_msgs)
        assert "You MUST call the search tool" in all_system_text

        # 2 Ollama calls (tool call + final)
        assert len(mock_ollama.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_response,expected_clean",
    [
        (
            "<think>Outline steps for planting tomato.</think>\nSure! Here are the steps:\n"
            "1. Choose a location",
            "Sure! Here are the steps:\n1. Choose a location",
        ),
        (
            "<thinking>internal reasoning goes here</thinking>The actual answer for the user.",
            "The actual answer for the user.",
        ),
        (
            "<THINK>case insensitive thinking</THINK>Clean response.",
            "Clean response.",
        ),
    ],
    ids=["think-tag", "thinking-tag", "think-uppercase"],
)
async def test_thinking_tags_not_leaked_to_user(
    raw_response,
    expected_clean,
    signal_server,
    mock_ollama,
    test_config,
    _mock_search,
    test_user_info,
    running_penny,
):
    """
    Regression test for #442: thinking tags embedded in content must be stripped.

    Some models (e.g. DeepSeek-style reasoning models) emit <think>...</think>
    blocks directly in the content field. These must not be forwarded to the user.
    """

    def handler(request, count):
        return mock_ollama._make_text_response(request, raw_response)

    mock_ollama.set_response_handler(handler)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what are the steps to plant a tomato?",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert expected_clean in response["message"], (
            f"Thinking tag content leaked to user. Got: {response['message']!r}"
        )
        assert "<think>" not in response["message"].lower()
        assert "<thinking>" not in response["message"].lower()


@pytest.mark.asyncio
async def test_entity_context_graceful_on_embed_failure(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """
    When the embed call fails, entity context is skipped and
    the normal search flow proceeds.
    """
    config = make_config(ollama_embedding_model="test-embed-model")

    def embed_handler(model, input_text):
        raise RuntimeError("embed service unavailable")

    mock_ollama.set_embed_handler(embed_handler)
    mock_ollama.set_default_flow(
        search_query="test query",
        final_response="search result! 🔍",
    )

    async with running_penny(config) as penny:
        # Seed entity with embedding
        entity = penny.db.get_or_create_entity(TEST_SENDER, "test entity")
        assert entity is not None and entity.id is not None
        penny.db.add_fact(entity.id, "some fact")
        penny.db.update_entity_embedding(entity.id, serialize_embedding([1.0, 0.0, 0.0, 0.0]))

        await signal_server.push_message(sender=TEST_SENDER, content="tell me about test entity")
        response = await signal_server.wait_for_message(timeout=10.0)

        # Falls back to normal search behavior
        assert "search result" in response["message"].lower()
        assert len(mock_ollama.requests) == 2
