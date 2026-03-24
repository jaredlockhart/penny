"""Integration tests for NotifyAgent."""

from datetime import UTC, datetime

import pytest

from penny.agents.notify import NotifyAgent, ThoughtMode
from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER, wait_until


def _seed_notify(penny):
    """Seed data needed for notifications: message, preference, thought."""
    penny.db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
    )
    pref = penny.db.preferences.add(
        user=TEST_SENDER,
        content="quantum computing",
        valence="positive",
    )
    penny.db.thoughts.add(
        TEST_SENDER,
        "I've been thinking about quantum computing",
        preference_id=pref.id if pref else None,
    )


# ── Eligibility checks ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_blocked_when_no_channel(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Notification is blocked when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_notify(penny)
        penny.notify_agent._channel = None
        assert not penny.notify_agent._should_notify(TEST_SENDER)


@pytest.mark.asyncio
async def test_notify_blocked_when_muted(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Notification is blocked when user is muted."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_notify(penny)
        penny.db.users.set_muted(TEST_SENDER)
        assert not penny.notify_agent._should_notify(TEST_SENDER)


@pytest.mark.asyncio
async def test_notify_blocked_when_no_thoughts(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Notification is blocked when user has no un-notified thoughts."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )
        assert not penny.notify_agent._should_notify(TEST_SENDER)


@pytest.mark.asyncio
async def test_notify_eligible_with_thoughts_and_channel(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Notification is eligible when all conditions are met."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_notify(penny)
        assert penny.notify_agent._should_notify(TEST_SENDER)


# ── Cooldown ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cooldown_elapsed_when_no_prior_autonomous(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Cooldown is always elapsed when no prior autonomous messages exist."""
    config = make_config()

    async with running_penny(config) as penny:
        assert penny.notify_agent._cooldown_elapsed(TEST_SENDER)


# ── Notification send modes ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_notify_thought_candidate(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
    mock_serper_image,
):
    """Thought candidate generates and sends a message with image attachment."""
    config = make_config(notify_candidates=1, serper_api_key="test-key")

    # Force thought candidate path (not checkin, not news)
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.99)

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(
            request, "hey, i was just thinking about quantum computing!"
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_notify(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        # Verify message was sent via the mock Signal server
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]
        assert "quantum" in response["message"].lower()

        # Thought should be marked as notified
        unnotified = penny.db.thoughts.get_next_unnotified(TEST_SENDER)
        assert unnotified is None

        # Serper image search should have been called with the message content
        mock_serper_image.assert_called_once()
        image_query = mock_serper_image.call_args[0][0]
        assert "quantum computing" in image_query.lower()
        assert response.get("base64_attachments"), "Notification should include an image"

        # Full system prompt structure assertion
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

## Identity
You are Penny. You and the user are friends who text regularly. \
This is mid-conversation — not a fresh chat.

Voice:
- Reply like you're continuing a text thread. No greetings, no sign-offs.
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

### Your Latest Thought
I've been thinking about quantum computing

## Instructions
You are reaching out to a friend proactively — sharing something \
interesting you've been thinking about or found in the news.

You have tools available:
- **search**: Search the web for current information. \
Accepts up to 5 queries per call.
Only call tools from the list above — do not call any other tool.

If your context includes 'Your Latest Thought', that contains research \
you already did. Share what's in it — the thought IS the substance of \
your message. You can search to add a fresh angle or find a link, but \
avoid re-searching the same topic.

Lead with the interesting thing — jump straight into it like you're \
picking up a text thread. Focus on ONE topic per message.

Include a follow-up URL so the user can read more about what you tell them. \
Pull the URL from your thought context or search results.

Every fact and detail in your message must come from your context."""
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"


@pytest.mark.asyncio
async def test_send_notify_news(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
    mock_serper_image,
):
    """News mode generates and sends a news message with image."""
    config = make_config(serper_api_key="test-key")

    # Force news path (not checkin)
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.0)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, "interesting news: **AI Breakthrough** changes everything!"
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_notify(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]
        assert response["message"]

        # Image search should use the first bold headline
        mock_serper_image.assert_called_once()
        image_query = mock_serper_image.call_args[0][0]
        assert image_query == "AI Breakthrough"
        assert response.get("base64_attachments"), "News should include an image"


@pytest.mark.asyncio
async def test_send_notify_checkin(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
    mock_serper_image,
):
    """Check-in sends a message with cat meme image."""
    config = make_config(serper_api_key="test-key")

    def handler(request, count):
        return mock_ollama._make_text_response(request, "hey! what have you been up to?")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_notify(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: True)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]
        assert response["message"]

        # Image search should use the check-in image prompt config
        mock_serper_image.assert_called_once()
        image_query = mock_serper_image.call_args[0][0]
        assert image_query == "funny cat meme"
        assert response.get("base64_attachments"), "Check-in should include an image"


@pytest.mark.asyncio
async def test_image_uses_content_when_no_image_prompt(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
    mock_serper_image,
):
    """Free-thinking thought (no seed topic) uses message content for image search."""
    config = make_config(notify_candidates=1, serper_api_key="test-key")

    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.99)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, "hey, did you know nitrous oxide lifetimes are shrinking?"
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed a free-thinking thought (no preference_id)
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
        )
        penny.db.thoughts.add(TEST_SENDER, "Thinking about atmospheric chemistry")
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]

        # With no search query and no seed topic, image_prompt is empty.
        # Channel should use content fallback (message text).
        mock_serper_image.assert_called_once()
        image_query = mock_serper_image.call_args[0][0]
        assert "nitrous" in image_query.lower()
        assert len(image_query) <= 300
        assert response.get("base64_attachments"), "Content fallback should produce an image"


# ── Image prompt extraction ──────────────────────────────────────────────


def test_extract_search_query_from_search_tool():
    """_extract_search_query extracts query from search tool calls."""
    from penny.agents.models import ToolCallRecord

    records = [
        ToolCallRecord(tool="search", arguments={"queries": ["latest space news"]}),
    ]
    result = NotifyAgent._extract_search_query(records)
    assert result == "latest space news"


def test_extract_search_query_returns_none_when_empty():
    """_extract_search_query returns None when no relevant tool calls."""
    result = NotifyAgent._extract_search_query([])
    assert result is None


def test_extract_first_headline():
    """_extract_first_headline extracts the first bold text from markdown."""
    text = "Hey! Here's the news:\n- **Big Story Title** - something happened\n- **Another One**"
    result = NotifyAgent._extract_first_headline(text)
    assert result == "Big Story Title"


def test_extract_first_headline_returns_none_when_no_bold():
    """_extract_first_headline returns None when no bold text present."""
    result = NotifyAgent._extract_first_headline("No bold text here")
    assert result is None


# ── Novelty scoring ──────────────────────────────────────────────────────


def test_novelty_score_full_when_no_recent():
    """Novelty is 1.0 when there are no recent messages to compare against."""
    from penny.ollama.similarity import novelty_score

    score = novelty_score([1.0, 0.0, 0.0], [])
    assert score == 1.0


def test_novelty_score_low_when_identical():
    """Novelty is low when candidate matches a recent message exactly."""
    from penny.ollama.similarity import novelty_score

    vec = [1.0, 0.0, 0.0]
    recent = [[1.0, 0.0, 0.0]]
    score = novelty_score(vec, recent)
    assert score < 0.01  # Nearly zero


# ── Thought context variants ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_thought_context_shows_notified_only(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """ChatAgent chat mode only shows thoughts that have been shared with the user."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.thoughts.add(TEST_SENDER, "shared thought about cats")
        penny.db.thoughts.add(TEST_SENDER, "unshared thought about dogs")

        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        cat_thought = [t for t in thoughts if "cats" in t.content][0]
        penny.db.thoughts.mark_notified(cat_thought.id)

        context = penny.chat_agent._thought_section(TEST_SENDER)
        if context:
            assert "cats" in context
            assert "dogs" not in context


@pytest.mark.asyncio
async def test_notify_thought_context_shows_specific_thought(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """NotifyAgent shows the specific thought being shared."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.thoughts.add(TEST_SENDER, "thinking about black holes")
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=1)

        penny.notify_agent._pending_thought = thoughts[0]
        context = penny.notify_agent._pending_thought_section()
        assert "black holes" in context
        assert "Your Latest Thought" in context

        # Candidate prompt includes thought but excludes conversation history
        now = datetime.now(UTC)
        penny.db.history.add(
            TEST_SENDER, now, now, PennyConstants.HistoryDuration.DAILY, "space games"
        )
        mode = ThoughtMode(thoughts[0])
        mode.prepare(penny.notify_agent)
        candidate_prompt = mode.build_system_prompt(penny.notify_agent, TEST_SENDER)
        assert "black holes" in candidate_prompt
        assert "Conversation History" not in candidate_prompt

        penny.notify_agent._pending_thought = None


# ── Candidate disqualification ────────────────────────────────────────


@pytest.mark.asyncio
async def test_disqualified_candidates_excluded(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Error fallbacks and model refusals are excluded from candidates."""
    config = make_config(notify_candidates=1)

    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.99)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, "Sorry, I couldn't complete that request within the allowed steps."
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_notify(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is False
        assert len(signal_server.outgoing_messages) == 0


def test_is_disqualified_error_strings():
    """Known error strings are disqualified."""
    from penny.responses import PennyResponse

    assert NotifyAgent._is_disqualified(PennyResponse.AGENT_MAX_STEPS)
    assert NotifyAgent._is_disqualified(PennyResponse.AGENT_MODEL_ERROR)
    assert NotifyAgent._is_disqualified(PennyResponse.FALLBACK_RESPONSE)


def test_is_disqualified_model_refusals():
    """Model refusal phrases are disqualified."""
    assert NotifyAgent._is_disqualified("I'm sorry, I can't help with that.")
    assert NotifyAgent._is_disqualified("I cannot do that right now.")
    assert NotifyAgent._is_disqualified("As an AI, I don't have personal thoughts.")
    assert NotifyAgent._is_disqualified("I apologize, but I'm unable to respond.")


def test_is_disqualified_allows_normal_messages():
    """Normal conversational messages are not disqualified."""
    assert not NotifyAgent._is_disqualified("Hey! Been thinking about quantum computing.")
    assert not NotifyAgent._is_disqualified("Check out this cool new game!")


# ── Tools-unavailable notification ──────────────────────────────────────


@pytest.mark.asyncio
async def test_news_tools_unavailable_sends_message(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """News mode sends tools-unavailable message when the news API fails."""
    from penny.responses import PennyResponse

    config = make_config()

    # Force news path
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.0)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, PennyResponse.AGENT_TOOLS_UNAVAILABLE.format(tools="fetch_news")
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_notify(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        msg = signal_server.outgoing_messages[-1]["message"]
        assert "wasn't able to get results" in msg
        assert "fetch_news" in msg


@pytest.mark.asyncio
async def test_thought_tools_unavailable_sends_message(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Thought candidate sends tools-unavailable message when search API fails."""
    from penny.responses import PennyResponse

    config = make_config(notify_candidates=1)

    # Force thought candidate path
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.99)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, PennyResponse.AGENT_TOOLS_UNAVAILABLE.format(tools="search")
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_notify(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        msg = signal_server.outgoing_messages[-1]["message"]
        assert "wasn't able to get results" in msg
        assert "search" in msg
