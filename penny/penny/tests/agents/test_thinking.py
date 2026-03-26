"""Integration tests for ThinkingAgent: continuous inner monologue loop.

Test organization:
1. Full integration (happy path) — seeded, free, news browse
2. Special success cases — rotation, threshold, rebuild
3. Error / edge cases — empty, duplicate, short report
4. URL validation — unit tests for hallucinated URL detection
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from penny.constants import PennyConstants
from penny.prompts import Prompt
from penny.tests.conftest import TEST_SENDER


async def _fake_embed(vec):
    """Return a fixed embedding vector (mock for embed_text)."""
    return vec


# Mock summary report long enough to pass MIN_THOUGHT_WORDS validation
MOCK_REPORT = (
    "Research report on AI topics. Recent developments include new model architectures, "
    "improved training techniques, and expanded deployment across industries. Key findings "
    "show significant progress in reasoning capabilities, multimodal understanding, and "
    "efficient inference. Several major organizations announced new initiatives in open "
    "source AI development and safety research during the past quarter."
)


def _seed_thinking(penny):
    """Seed a message (so user exists), history, and preference (so seed topic exists)."""
    penny.db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello penny"
    )
    penny.db.history.add(
        user=TEST_SENDER,
        period_start=datetime(2026, 3, 3),
        period_end=datetime(2026, 3, 4),
        duration=PennyConstants.HistoryDuration.DAILY,
        topics="- Quantum gravity experiments\n- Cyberpunk anime releases",
    )
    penny.db.preferences.add(
        user=TEST_SENDER,
        content="Quantum gravity experiments",
        valence="positive",
        source=PennyConstants.PreferenceSource.MANUAL,
    )


def _add_dislike(penny):
    """Add a negative preference for context assertions."""
    penny.db.preferences.add(
        user=TEST_SENDER,
        content="Country music",
        valence="negative",
    )


# ── 1. Full integration (happy path) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_seeded_thinking_full_loop(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Seeded thinking: full multi-step loop with tools, context, dedup, and storage."""
    # Force seeded path: 0% free/news → always seeded
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    monkeypatch.setattr("penny.agents.thinking.random.choice", lambda lst: lst[0])
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=3,
        free_thinking_probability=0.0,
        news_thinking_probability=0.0,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            # Step 1: tool call
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["quantum gravity 2026"], "reasoning": "Researching seed topic"},
            )
        if count == 2:
            # Step 2: text reflecting on search results
            return mock_ollama._make_text_response(
                request, "Found interesting quantum gravity results from the search."
            )
        if count == 3:
            # Step 3: more text
            return mock_ollama._make_text_response(request, "The implications are fascinating.")
        # Summary call
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        _add_dislike(penny)

        # Pre-seed a thought with matching preference_id for context
        penny.db.thoughts.add(TEST_SENDER, "Previous thought about gravity", preference_id=1)

        await penny.thinking_agent.execute()

        # -- Request count: 3 thinking steps + 1 summary = 4
        assert len(requests_seen) == 4

        # -- Prompt: seed topic drives the user message
        first_user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        assert "Quantum gravity experiments" in first_user_msgs[0]["content"]

        # -- Full system prompt structure assertion
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        # Strip dynamic timestamp line, verify everything else exactly
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

## Context
### Recent Background Thinking
Previous thought about gravity

### Topics to Avoid
- Country music

## Instructions
You are thinking to yourself. This is your inner monologue — \
the user cannot see this.

Your goal is to find ONE specific, concrete thing worth knowing about — \
something the user would enjoy hearing about. Look for new releases, \
creative work, technical deep-dives, or discoveries. Avoid \
troubleshooting guides, bug reports, and support articles.

You have tools available:
- **search**: Search the web for current information. \
Accepts up to 1 query per call.

Go DEEP, not wide:
- Search for the topic, then pick the single most interesting result
- Do follow-up searches to learn more about that specific thing
- Do NOT search for a different subtopic on each step
- Do NOT repeat the same search query you already ran

When you receive 'dig deeper', that means: learn more about what \
you already found. More detail on the same thing, not a new thing.

Check your recent thoughts to avoid repeating what you already explored.

All information in your responses must come from your tool results. \
If nothing interesting comes up, that's fine — quiet cycles are normal."""
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"

        # -- Tool results flow into subsequent steps
        step2_msgs = requests_seen[1]["messages"]
        tool_msgs = [m for m in step2_msgs if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

        # -- "dig deeper" continuation injected between text steps
        step3_msgs = requests_seen[2]["messages"]
        user_msgs = [m for m in step3_msgs if m.get("role") == "user"]
        assert any(m.get("content") == "dig deeper into what you just found" for m in user_msgs)

        # -- Tools available on final agentic step (not stripped for thinking)
        last_loop_request = requests_seen[2]
        last_loop_tools = last_loop_request.get("tools") or []
        assert len(last_loop_tools) > 0, "Final step should keep tools for thinking agent"

        # -- Summary input contains tool results, not model text
        summary_request = requests_seen[3]
        summary_user_msg = [m for m in summary_request["messages"] if m.get("role") == "user"][0][
            "content"
        ]
        assert "Mock search results" in summary_user_msg  # raw tool output
        assert "Found interesting" not in summary_user_msg  # not model text

        # -- Storage: summary stored (not raw monologue), with correct preference_id
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        stored = [t for t in thoughts if t.content != "Previous thought about gravity"]
        assert len(stored) == 1
        assert "Research report" in stored[0].content
        assert stored[0].preference_id == 1
        assert "Found interesting" not in stored[0].content  # raw monologue not stored

        # -- Preference marked as thought-about
        pool = penny.db.preferences.get_least_recent_positive(TEST_SENDER)
        assert any(p.last_thought_at is not None for p in pool)


@pytest.mark.asyncio
async def test_free_thinking_full_loop(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Free thinking: identical loop to seeded — context, tools, dedup, storage."""
    # Force free-thinking path: 100% free → always free
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.0)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=3,
        free_thinking_probability=1.0,
        news_thinking_probability=0.0,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["interesting science 2026"], "reasoning": "Exploring freely"},
            )
        if count == 2:
            return mock_ollama._make_text_response(
                request, "Found something interesting about biology."
            )
        if count == 3:
            return mock_ollama._make_text_response(request, "This is a novel finding.")
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        _add_dislike(penny)

        # Pre-seed thoughts: one free + one seeded so free is underrepresented
        # relative to FREE_THINKING_PROBABILITY=1.0 (actual 50% < target 100%)
        pref = penny.db.preferences.get_least_recent_positive(TEST_SENDER)[0]
        penny.db.thoughts.add(TEST_SENDER, "Previous free thought about space", preference_id=None)
        penny.db.thoughts.add(TEST_SENDER, "Previous seeded thought", preference_id=pref.id)

        await penny.thinking_agent.execute()

        # -- Request count: 3 thinking steps + 1 summary = 4
        assert len(requests_seen) == 4

        # -- Prompt: free exploration prompt (not seed-based)
        first_user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        first_prompt = first_user_msgs[0]["content"].lower()
        assert "search for something" in first_prompt or "interesting" in first_prompt

        # -- Full system prompt: no identity, no profile, no free thoughts, just dislikes
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

## Context
### Topics to Avoid
- Country music

## Instructions
You are thinking to yourself. This is your inner monologue — \
the user cannot see this.

Your goal is to find ONE specific, concrete thing worth knowing about — \
something the user would enjoy hearing about. Look for new releases, \
creative work, technical deep-dives, or discoveries. Avoid \
troubleshooting guides, bug reports, and support articles.

You have tools available:
- **search**: Search the web for current information. \
Accepts up to 1 query per call.

Go DEEP, not wide:
- Search for the topic, then pick the single most interesting result
- Do follow-up searches to learn more about that specific thing
- Do NOT search for a different subtopic on each step
- Do NOT repeat the same search query you already ran

When you receive 'dig deeper', that means: learn more about what \
you already found. More detail on the same thing, not a new thing.

Check your recent thoughts to avoid repeating what you already explored.

All information in your responses must come from your tool results. \
If nothing interesting comes up, that's fine — quiet cycles are normal."""
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"

        # -- Tools: search available, message_user absent
        tools = requests_seen[0].get("tools") or []
        tool_names = [t["function"]["name"] for t in tools]
        assert "search" in tool_names
        assert "message_user" not in tool_names

        # -- Tool results flow into subsequent steps
        step2_msgs = requests_seen[1]["messages"]
        tool_msgs = [m for m in step2_msgs if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

        # -- "dig deeper" continuation injected between text steps
        step3_msgs = requests_seen[2]["messages"]
        user_msgs = [m for m in step3_msgs if m.get("role") == "user"]
        assert any(m.get("content") == "dig deeper into what you just found" for m in user_msgs)

        # -- Storage: summary stored with preference_id=None
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        pre_seeded = {"Previous free thought about space", "Previous seeded thought"}
        stored = [t for t in thoughts if t.content not in pre_seeded]
        assert len(stored) == 1
        assert "Research report" in stored[0].content
        assert stored[0].preference_id is None
        assert "Found something" not in stored[0].content  # raw monologue not stored

        # -- No preference marked (free thinking has no seed preference)
        pool = penny.db.preferences.get_least_recent_positive(TEST_SENDER)
        # The preference from _seed_thinking should still have NULL last_thought_at
        assert all(p.last_thought_at is None for p in pool)


@pytest.mark.asyncio
async def test_news_thinking_full_loop(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """News thinking: intentional news mode — reads news, picks a story, digs in."""
    # Force news path: 0% pure-free, 100% news → _pick_free_prompt always picks news
    # random.random used for: (1) empty-pool fallback in _should_think_free,
    # (2) news/free coin flip in _pick_free_prompt — 0.0 < 1.0 → news wins
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.0)
    monkeypatch.setattr("penny.tools.news.NewsTool.search", AsyncMock(return_value=[]))
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
        news_api_key="fake-key",
        free_thinking_probability=0.0,
        news_thinking_probability=1.0,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["top news stories 2026"], "reasoning": "Reading the news"},
            )
        if count == 2:
            return mock_ollama._make_text_response(request, "Found a compelling story.")
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        _add_dislike(penny)

        await penny.thinking_agent.execute()

        # Should run and produce a thought
        assert len(requests_seen) > 0
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 1

        # Prompt should be the news thinking prompt
        first_user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        assert "news" in first_user_msgs[0]["content"].lower()

        # -- Full system prompt: no identity, no profile, no thoughts, just dislikes
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

## Context
### Topics to Avoid
- Country music

## Instructions
You are thinking to yourself. This is your inner monologue — \
the user cannot see this.

Your goal is to find ONE specific, concrete thing worth knowing about — \
something the user would enjoy hearing about. Look for new releases, \
creative work, technical deep-dives, or discoveries. Avoid \
troubleshooting guides, bug reports, and support articles.

You have tools available:
- **search**: Search the web for current information. \
Accepts up to 1 query per call.
- **fetch_news**: Search for recent news articles on a topic. \
Returns headlines, summaries, and URLs.

Go DEEP, not wide:
- Search for the topic, then pick the single most interesting result
- Do follow-up searches to learn more about that specific thing
- Do NOT search for a different subtopic on each step
- Do NOT repeat the same search query you already ran

When you receive 'dig deeper', that means: learn more about what \
you already found. More detail on the same thing, not a new thing.

Check your recent thoughts to avoid repeating what you already explored.

All information in your responses must come from your tool results. \
If nothing interesting comes up, that's fine — quiet cycles are normal."""
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"

        # -- Tools: both search and fetch_news available
        tools = requests_seen[0].get("tools") or []
        tool_names = [t["function"]["name"] for t in tools]
        assert "search" in tool_names
        assert "fetch_news" in tool_names

        # No preference marked (news thinking has no seed preference)
        pool = penny.db.preferences.get_least_recent_positive(TEST_SENDER)
        assert all(p.last_thought_at is None for p in pool)


@pytest.mark.asyncio
async def test_news_browsing_full_loop(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """News browsing fallback: when no preferences exist, browses news and stores thought."""
    # Force seeded path so it hits "no preferences → browse news"
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
        free_thinking_probability=0.0,
        news_thinking_probability=0.0,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["latest news 2026"], "reasoning": "Browsing news"},
            )
        if count == 2:
            # Final step — forced text (ignored, search results summarized in after_run)
            return mock_ollama._make_text_response(request, "Found some news.")
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # User exists but no preferences — triggers news browsing
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )

        await penny.thinking_agent.execute()

        # Should run and produce a thought
        assert len(requests_seen) > 0
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 1

        # Prompt should be the news browsing prompt
        first_user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        assert any("news" in m.get("content", "").lower() for m in first_user_msgs)


# ── 2. Special success cases ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preference_rotation_via_last_thought_at(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """After thinking about a preference, it rotates to the back of the pool."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    monkeypatch.setattr("penny.agents.thinking.random.choice", lambda lst: lst[0])
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
        free_thinking_probability=0.0,
        news_thinking_probability=0.0,
    )

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["astrophysics 2026"], "reasoning": "Researching"},
            )
        if count == 2:
            return mock_ollama._make_text_response(request, "ok")
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )
        pref_a = penny.db.preferences.add(
            user=TEST_SENDER,
            content="astrophysics",
            valence="positive",
            source=PennyConstants.PreferenceSource.MANUAL,
        )
        pref_b = penny.db.preferences.add(
            user=TEST_SENDER,
            content="cyberpunk anime",
            valence="positive",
            source=PennyConstants.PreferenceSource.MANUAL,
        )
        assert pref_a is not None and pref_b is not None

        await penny.thinking_agent.execute()

        # The used preference has last_thought_at set; the other doesn't
        updated = penny.db.preferences.get_least_recent_positive(TEST_SENDER, pool_size=5)
        thought_about = [p for p in updated if p.last_thought_at is not None]
        not_thought = [p for p in updated if p.last_thought_at is None]
        assert len(thought_about) == 1
        assert len(not_thought) == 1

        # Un-thought-about preference sorts first in the pool
        pool = penny.db.preferences.get_least_recent_positive(TEST_SENDER, pool_size=5)
        assert pool[0].last_thought_at is None


@pytest.mark.asyncio
async def test_extracted_preference_below_threshold_skipped(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Extracted preferences below mention threshold are not used as seeds."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
        free_thinking_probability=0.0,
        news_thinking_probability=0.0,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )
        # mention_count=1, below default threshold of 2
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="casual topic",
            valence="positive",
            source=PennyConstants.PreferenceSource.EXTRACTED,
            mention_count=1,
        )

        await penny.thinking_agent.execute()

        # Falls back to news browse (no eligible seed)
        user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        assert "casual topic" not in user_msgs[0]["content"]


@pytest.mark.asyncio
async def test_extracted_preference_at_threshold_used(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Extracted preferences at/above mention threshold ARE used as seeds."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
        free_thinking_probability=0.0,
        news_thinking_probability=0.0,
    )

    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hello"
        )
        # mention_count=3, meets default threshold of 2
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="repeated interest topic",
            valence="positive",
            source=PennyConstants.PreferenceSource.EXTRACTED,
            mention_count=3,
        )

        await penny.thinking_agent.execute()

        user_msgs = [m for m in requests_seen[0]["messages"] if m.get("role") == "user"]
        assert "repeated interest topic" in user_msgs[0]["content"]


@pytest.mark.asyncio
async def test_scheduler_runs_history_before_thinking(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """History is scheduled before thinking so context is fresh."""
    config = make_config()

    async with running_penny(config) as penny:
        schedules = penny.scheduler._schedules
        agent_names = [s.agent.name for s in schedules]

        history_idx = agent_names.index("history")
        thinking_idx = agent_names.index("inner_monologue")
        assert history_idx < thinking_idx


@pytest.mark.asyncio
async def test_distribution_steers_toward_underrepresented_type(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
):
    """When free thoughts are underrepresented, thinking picks free; vice versa."""
    # Target: 50% free, 50% seeded
    config = make_config(
        inner_monologue_interval=99999.0,
        free_thinking_probability=0.5,
        news_thinking_probability=0.0,
    )

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        pref = penny.db.preferences.get_least_recent_positive(TEST_SENDER)[0]

        # All seeded, 0 free → free is underrepresented → should pick free
        for i in range(3):
            penny.db.thoughts.add(TEST_SENDER, f"seeded {i}", preference_id=pref.id)
        prompt = await penny.thinking_agent.get_prompt(TEST_SENDER)
        assert prompt is not None
        assert prompt == Prompt.THINKING_FREE

        # Reset: all free, 0 seeded → seeded is underrepresented → should pick seeded
        for t in penny.db.thoughts.get_all_unnotified(TEST_SENDER):
            penny.db.thoughts.mark_notified(t.id)
        for i in range(3):
            penny.db.thoughts.add(TEST_SENDER, f"free {i}")
        penny.thinking_agent._seed_topic = None
        penny.thinking_agent._seed_pref_id = None
        prompt = await penny.thinking_agent.get_prompt(TEST_SENDER)
        assert prompt is not None
        assert prompt != Prompt.THINKING_FREE


@pytest.mark.asyncio
async def test_thinking_skips_when_too_many_unnotified(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """Thinking agent skips cycle when unnotified thoughts reach the cap."""
    config = make_config(
        inner_monologue_interval=99999.0,
        max_unnotified_thoughts=3,
    )

    async with running_penny(config) as penny:
        _seed_thinking(penny)

        # Add thoughts up to the cap
        for i in range(3):
            penny.db.thoughts.add(TEST_SENDER, f"unnotified thought {i}")

        # execute returns True (skipped but signals "I ran" to reset timer)
        initial_count = len(penny.db.thoughts.get_all_unnotified(TEST_SENDER))
        result = await penny.thinking_agent.execute()
        assert result is True
        # No new thoughts were added (skipped, not executed)
        assert len(penny.db.thoughts.get_all_unnotified(TEST_SENDER)) == initial_count

        # Notifying one thought should allow thinking to proceed again
        thought = penny.db.thoughts.get_next_unnotified(TEST_SENDER)
        penny.db.thoughts.mark_notified(thought.id)
        assert penny.db.thoughts.count_unnotified(TEST_SENDER) == 2

        # Now thinking should generate a prompt (not skip)
        prompt = await penny.thinking_agent.get_prompt(TEST_SENDER)
        assert prompt is not None


# ── 3. Error / edge cases ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_monologue_skips_storage(
    signal_server, mock_ollama, make_config, _mock_search, test_user_info, running_penny
):
    """No thought stored when the model produces empty content."""
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=1,
    )

    def handler(request, count):
        return mock_ollama._make_text_response(request, "")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        await penny.thinking_agent.execute()

        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 0


@pytest.mark.asyncio
async def test_seeded_duplicate_thought_skips_storage(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Seeded: when a new thought is too similar to existing, it is not stored."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    monkeypatch.setattr("penny.agents.thinking.random.choice", lambda lst: lst[0])
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
        free_thinking_probability=0.0,
        news_thinking_probability=0.0,
    )

    duplicate_vec = [1.0, 0.0, 0.0]
    monkeypatch.setattr(
        "penny.agents.thinking.embed_text",
        lambda _client, _text: _fake_embed(duplicate_vec),  # noqa: ARG005
    )

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["quantum gravity"], "reasoning": "Researching"},
            )
        if count == 2:
            return mock_ollama._make_text_response(request, "Yep, same old stuff.")
        return mock_ollama._make_text_response(
            request, "Confirmed the album still exists, nothing new."
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        penny.thinking_agent._embedding_model_client = object()

        penny.db.thoughts.add(TEST_SENDER, "Old thought about the same topic.", preference_id=1)

        await penny.thinking_agent.execute()

        # Only the pre-seeded thought remains
        thoughts = penny.db.thoughts.get_recent_by_preference(TEST_SENDER, preference_id=1)
        assert len(thoughts) == 1
        assert "Old thought" in thoughts[0].content

        # Preference still marked as thought-about
        pool = penny.db.preferences.get_least_recent_positive(TEST_SENDER)
        assert any(p.last_thought_at is not None for p in pool)


@pytest.mark.asyncio
async def test_free_duplicate_thought_skips_storage(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """Free: when a new thought is too similar to existing free thought, it is not stored."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.0)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
        free_thinking_probability=1.0,
        news_thinking_probability=0.0,
    )

    duplicate_vec = [1.0, 0.0, 0.0]
    monkeypatch.setattr(
        "penny.agents.thinking.embed_text",
        lambda _client, _text: _fake_embed(duplicate_vec),  # noqa: ARG005
    )

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["interesting science"], "reasoning": "Exploring"},
            )
        if count == 2:
            return mock_ollama._make_text_response(request, "Yep, same old stuff.")
        return mock_ollama._make_text_response(
            request, "Confirmed quantum computers still quantum, nothing new."
        )

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        penny.thinking_agent._embedding_model_client = object()

        penny.db.thoughts.add(
            TEST_SENDER, "Old free thought about quantum stuff.", preference_id=None
        )

        await penny.thinking_agent.execute()

        # Only the pre-seeded free thought remains
        free_thoughts = penny.db.thoughts.get_recent_by_preference(TEST_SENDER, preference_id=None)
        assert len(free_thoughts) == 1
        assert "Old free thought" in free_thoughts[0].content


@pytest.mark.asyncio
async def test_novel_thought_is_stored(
    signal_server,
    mock_ollama,
    make_config,
    _mock_search,
    test_user_info,
    running_penny,
    monkeypatch,
):
    """When a new thought is sufficiently different from existing ones, it is stored."""
    monkeypatch.setattr("penny.agents.thinking.random.random", lambda: 0.99)
    config = make_config(
        inner_monologue_interval=99999.0,
        inner_monologue_max_steps=2,
        free_thinking_probability=0.0,
        news_thinking_probability=0.0,
    )

    call_count = 0

    async def _alternating_embed(_client, _text):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr("penny.agents.thinking.embed_text", _alternating_embed)

    def handler(request, count):
        if count == 1:
            return mock_ollama._make_tool_call_response(
                request,
                "search",
                {"queries": ["quantum gravity"], "reasoning": "Researching"},
            )
        if count == 2:
            return mock_ollama._make_text_response(request, "Found something new!")
        return mock_ollama._make_text_response(request, MOCK_REPORT)

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_thinking(penny)
        penny.thinking_agent._embedding_model_client = object()

        penny.db.thoughts.add(TEST_SENDER, "Old thought about a different topic.", preference_id=1)

        await penny.thinking_agent.execute()

        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        assert len(thoughts) == 2


# ── 4. URL validation (unit tests) ───────────────────────────────────────


class TestSummaryUrlValidation:
    """Test that hallucinated URLs are detected in thought summaries."""

    def test_valid_urls_pass(self):
        """URLs that appear in source text are not flagged."""
        from penny.agents.thinking import ThinkingAgent

        source = "Check out https://example.com/article and https://arxiv.org/abs/1234"
        report = "Found this: https://example.com/article"
        assert ThinkingAgent._find_hallucinated_urls(report, source) == []

    def test_hallucinated_urls_detected(self):
        """URLs not in source text are flagged."""
        from penny.agents.thinking import ThinkingAgent

        source = "Check out https://example.com/real-article for details."
        report = "Read more: https://example.com/fake-article"
        bad = ThinkingAgent._find_hallucinated_urls(report, source)
        assert len(bad) == 1
        assert "fake-article" in bad[0]

    def test_no_urls_passes(self):
        """Report with no URLs passes validation."""
        from penny.agents.thinking import ThinkingAgent

        assert ThinkingAgent._find_hallucinated_urls("No links here.", "some source") == []

    def test_markdown_urls_checked(self):
        """URLs inside markdown links are also validated."""
        from penny.agents.thinking import ThinkingAgent

        source = "Found at https://real.com/page"
        report = "See [this article](https://fake.com/page) for details."
        bad = ThinkingAgent._find_hallucinated_urls(report, source)
        assert len(bad) == 1
        assert "fake.com" in bad[0]

    def test_trailing_punctuation_stripped(self):
        """URLs with trailing punctuation still match source."""
        from penny.agents.thinking import ThinkingAgent

        source = "URL: https://example.com/article"
        report = "Check https://example.com/article."
        assert ThinkingAgent._find_hallucinated_urls(report, source) == []
