"""Integration tests for NotifyAgent proactive messaging."""

from datetime import UTC, datetime

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER, wait_until


def _seed_proactive(penny):
    """Seed data needed for proactive messaging: message, history, thought."""
    penny.db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
    )
    penny.db.thoughts.add(TEST_SENDER, "I've been thinking about quantum computing")


# ── Eligibility checks ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proactive_blocked_when_no_channel(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Proactive messaging is blocked when no channel is set."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_proactive(penny)
        penny.notify_agent._channel = None
        assert not penny.notify_agent._should_send_proactive(TEST_SENDER)


@pytest.mark.asyncio
async def test_proactive_blocked_when_muted(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Proactive messaging is blocked when user is muted."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_proactive(penny)
        penny.db.users.set_muted(TEST_SENDER)
        assert not penny.notify_agent._should_send_proactive(TEST_SENDER)


@pytest.mark.asyncio
async def test_proactive_blocked_when_no_thoughts(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Proactive messaging is blocked when user has no un-notified thoughts."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )
        assert not penny.notify_agent._should_send_proactive(TEST_SENDER)


@pytest.mark.asyncio
async def test_proactive_eligible_with_thoughts_and_channel(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Proactive messaging is eligible when all conditions are met."""
    config = make_config()

    async with running_penny(config) as penny:
        _seed_proactive(penny)
        assert penny.notify_agent._should_send_proactive(TEST_SENDER)


# ── Cooldown ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cooldown_elapsed_when_no_prior_autonomous(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Cooldown is always elapsed when no prior autonomous messages exist."""
    config = make_config()

    async with running_penny(config) as penny:
        assert penny.notify_agent._cooldown_elapsed(TEST_SENDER)


# ── Proactive send modes ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_proactive_thought_candidate(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Proactive thought candidate generates and sends a message."""
    config = make_config(proactive_candidates=1)

    # Force thought candidate path (not checkin, not news)
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.99)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, "hey, i was just thinking about quantum computing!"
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_proactive(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        # Verify message was sent via the mock Signal server
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]
        assert "quantum" in response["message"].lower()

        # Thought should be marked as notified
        unnotified = penny.db.thoughts.get_next_unnotified(TEST_SENDER, freshness_hours=24)
        assert unnotified is None


@pytest.mark.asyncio
async def test_send_proactive_news(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Proactive news mode generates and sends a news message."""
    config = make_config()

    # Force news path (not checkin)
    monkeypatch.setattr("penny.agents.notify.random.random", lambda: 0.0)

    def handler(request, count):
        return mock_ollama._make_text_response(
            request, "interesting news today about AI breakthroughs!"
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_proactive(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: False)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]
        assert response["message"]  # Non-empty response sent


@pytest.mark.asyncio
async def test_send_proactive_checkin(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Proactive check-in sends a message when conditions are met."""
    config = make_config()

    def handler(request, count):
        return mock_ollama._make_text_response(request, "hey! what have you been up to?")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_proactive(penny)
        monkeypatch.setattr(penny.notify_agent, "_should_checkin", lambda user: True)

        result = await penny.notify_agent.execute_for_user(TEST_SENDER)
        assert result is True

        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        response = signal_server.outgoing_messages[-1]
        assert response["message"]


# ── Image prompt extraction ──────────────────────────────────────────────


def test_extract_image_prompt_from_news_tool():
    """_extract_image_prompt extracts topic from fetch_news tool calls."""
    from penny.agents.models import ToolCallRecord

    records = [
        ToolCallRecord(tool="fetch_news", arguments={"topic": "AI breakthroughs"}),
    ]
    result = NotifyAgent._extract_image_prompt(records)
    assert result == "AI breakthroughs"


def test_extract_image_prompt_from_search_tool():
    """_extract_image_prompt extracts query from search tool calls."""
    from penny.agents.models import ToolCallRecord

    records = [
        ToolCallRecord(tool="search", arguments={"query": "latest space news"}),
    ]
    result = NotifyAgent._extract_image_prompt(records)
    assert result == "latest space news"


def test_extract_image_prompt_returns_none_when_empty():
    """_extract_image_prompt returns None when no relevant tool calls."""
    result = NotifyAgent._extract_image_prompt([])
    assert result is None


# ── Novelty scoring ──────────────────────────────────────────────────────


def test_novelty_score_full_when_no_recent():
    """Novelty is 1.0 when there are no recent messages to compare against."""
    score = NotifyAgent._novelty_score([1.0, 0.0, 0.0], [])
    assert score == 1.0


def test_novelty_score_low_when_identical():
    """Novelty is low when candidate matches a recent message exactly."""
    vec = [1.0, 0.0, 0.0]
    recent = [[1.0, 0.0, 0.0]]
    score = NotifyAgent._novelty_score(vec, recent)
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
async def test_proactive_thought_context_shows_specific_thought(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """NotifyAgent proactive mode shows the specific thought being shared."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.thoughts.add(TEST_SENDER, "thinking about black holes")
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=1)

        penny.notify_agent._proactive_thought = thoughts[0]
        context = penny.notify_agent._build_proactive_thought_context()
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

        penny.notify_agent._proactive_thought = None


# Need to import NotifyAgent for static method tests
from penny.agents.notify import NotifyAgent  # noqa: E402
