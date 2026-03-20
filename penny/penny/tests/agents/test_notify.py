"""Integration tests for NotifyAgent."""

from datetime import UTC, datetime

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER, wait_until


def _seed_notify(penny):
    """Seed data needed for notifications: message, history, thought."""
    penny.db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
    )
    penny.db.thoughts.add(TEST_SENDER, "I've been thinking about quantum computing")


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
):
    """Thought candidate generates and sends a message."""
    config = make_config(notify_candidates=1)

    # Force thought candidate path (not checkin, not news)
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.99)

    def handler(request, count):
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


@pytest.mark.asyncio
async def test_send_notify_news(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """News mode generates and sends a news message."""
    config = make_config()

    # Force news path (not checkin)
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.0)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, "interesting news today about AI breakthroughs!"
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_notify(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]
        assert response["message"]  # Non-empty response sent


@pytest.mark.asyncio
async def test_send_notify_checkin(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Check-in sends a message when conditions are met."""
    config = make_config()

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


# ── Image prompt extraction ──────────────────────────────────────────────


def test_extract_search_query_from_search_tool():
    """_extract_search_query extracts query from search tool calls."""
    from penny.agents.models import ToolCallRecord

    records = [
        ToolCallRecord(tool="search", arguments={"query": "latest space news"}),
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

        context = penny.chat_agent._build_thought_context(TEST_SENDER)
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
        context = penny.notify_agent._build_pending_thought_context()
        assert "black holes" in context
        assert "Your Latest Thought" in context

        # Candidate context includes thought but excludes conversation history
        now = datetime.now(UTC)
        penny.db.history.add(
            TEST_SENDER, now, now, PennyConstants.HistoryDuration.DAILY, "space games"
        )
        candidate_ctx = penny.notify_agent._build_thought_candidate_context(TEST_SENDER)
        assert "black holes" in candidate_ctx
        assert "Conversation History" not in candidate_ctx

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


# Need to import NotifyAgent for static method tests
from penny.agents.notify import NotifyAgent  # noqa: E402
