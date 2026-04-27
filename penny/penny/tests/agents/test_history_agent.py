"""Integration tests for HistoryAgent: preference and knowledge extraction.

Both flows are model-driven shells around the standard memory tool
surface.  Test organisation:

1. Preference extraction (agent shell) — happy path with verbatim
   system prompt + cursor advancement on success / no advancement on
   failure / empty log path
2. Knowledge extraction (agent shell) — happy path with verbatim
   system prompt + cursor advancement, plus the no-pages and
   max-steps shapes
"""

from __future__ import annotations

import pytest

from penny.agents.knowledge_extractor import KnowledgeExtractorAgent
from penny.agents.preference_extractor import PreferenceExtractorAgent
from penny.constants import PennyConstants
from penny.database.memory_store import LogEntryInput
from penny.tests.conftest import TEST_SENDER


def _seed_user_message(penny, content: str) -> None:
    """Seed an entry into the user-messages log (simulates channel ingress)."""
    penny.db.memories.append(
        PennyConstants.MEMORY_USER_MESSAGES_LOG,
        [LogEntryInput(content=content)],
        author="user",
    )


def _seed_browse_page(penny, url: str, title: str, body: str) -> None:
    """Seed one page section into the browse-results log (simulates BrowseTool)."""
    section = (
        f"{PennyConstants.BROWSE_PAGE_HEADER}{url}\n"
        f"{PennyConstants.BROWSE_TITLE_PREFIX}{title}\n"
        f"{PennyConstants.BROWSE_URL_PREFIX}{url}\n\n"
        f"{body}"
    )
    penny.db.memories.append(
        PennyConstants.MEMORY_BROWSE_RESULTS_LOG,
        [LogEntryInput(content=section)],
        author="chat",
    )


def _cursor_for_preference_extractor(penny):
    return penny.db.cursors.get(
        PreferenceExtractorAgent.name, PennyConstants.MEMORY_USER_MESSAGES_LOG
    )


def _cursor_for_knowledge_extractor(penny):
    return penny.db.cursors.get(
        KnowledgeExtractorAgent.name, PennyConstants.MEMORY_BROWSE_RESULTS_LOG
    )


# ── 1. Preference extraction (agent shell) ───────────────────────────────


@pytest.mark.asyncio
async def test_preference_extraction_full_loop(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Core success flow: read user-messages, write to likes, call done.

    Asserts the verbatim system prompt drives the loop, the model's
    ``collection_write`` tool call lands an entry in the ``likes`` memory,
    and the cursor advances so the next run sees no new messages.
    """
    config = make_config(history_interval=99999.0)
    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_llm._make_tool_call_response(
                request, "log_read_next", {"memory": "user-messages"}
            )
        if count == 2:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "likes",
                    "entries": [
                        {
                            "key": "single-origin coffee beans",
                            "content": "I really love single-origin coffee beans",
                        }
                    ],
                },
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_user_message(penny, "I really love single-origin coffee beans")

        result = await penny.preference_extractor_agent.execute_for_user(TEST_SENDER)

        assert result is True

        # Full exact system prompt the model saw on its first step
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

You extract the user's likes and dislikes from their recent messages.

1. Call log_read_next("user-messages") to fetch messages you haven't seen yet.
2. Identify every genuine preference across the returned messages.
3. Call collection_write once per target collection — likes for things \
the user wants/enjoys/seeks, dislikes for things they avoid/complain \
about — batching all entries.
4. Call done().

Each entry's key is a fully-qualified topic name (3-10 words, e.g. \
'Talk (album) by Yes', 'Dune Part Two (2024 film)') — NOT a vague \
phrase like 'the album'. The content is the user's raw message that \
expressed the preference.

Skip factual statements, questions, and troubleshooting requests. \
Only extract topics the USER expressed interest in — not Penny's \
opinions, not topics merely mentioned in passing. If a user is \
frustrated about NOT FINDING something they want, that's a like; \
negative means they dislike the thing itself.

If no preferences appear in the returned messages, just call done() \
without writing anything."""
        assert rest == expected, (
            f"Preference extractor system prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"
        )

        # Write landed in the likes collection, attributed to the extractor
        likes = penny.db.memories.read_all("likes")
        assert any(e.key == "single-origin coffee beans" for e in likes)
        coffee = next(e for e in likes if e.key == "single-origin coffee beans")
        assert coffee.author == PreferenceExtractorAgent.name

        # Cursor advanced past the seeded message
        cursor = _cursor_for_preference_extractor(penny)
        assert cursor is not None


@pytest.mark.asyncio
async def test_cursor_does_not_advance_on_max_steps(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """If the model exhausts max_steps without calling done, the cursor stays
    where it was so the next run sees the same messages again."""
    config = make_config(history_interval=99999.0)

    # Always return log_read_next — never done.  Loop hits max_steps.
    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_tool_call_response(
            request, "log_read_next", {"memory": "user-messages"}
        )
    )

    async with running_penny(config) as penny:
        _seed_user_message(penny, "first message")
        before = _cursor_for_preference_extractor(penny)

        result = await penny.preference_extractor_agent.execute_for_user(TEST_SENDER)

        assert result is False
        after = _cursor_for_preference_extractor(penny)
        assert after == before


@pytest.mark.asyncio
async def test_no_user_messages_completes_cleanly(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Empty user-messages log → model reads nothing, calls done immediately."""
    config = make_config(history_interval=99999.0)

    def handler(request, _count):
        # First step: read the (empty) log; subsequent step: call done.
        messages = request.get("messages", [])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            return mock_llm._make_tool_call_response(
                request, "log_read_next", {"memory": "user-messages"}
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        result = await penny.preference_extractor_agent.execute_for_user(TEST_SENDER)

        assert result is True
        assert penny.db.memories.read_all("likes") == []
        assert penny.db.memories.read_all("dislikes") == []


def test_preference_extractor_max_steps_constant_is_set():
    """Defensive check: the cap is non-zero so the loop can actually run."""
    assert PreferenceExtractorAgent.MAX_STEPS > 0


# ── 2. Knowledge extraction (agent shell) ────────────────────────────────


@pytest.mark.asyncio
async def test_knowledge_extraction_full_loop(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Core success flow: read browse-results, write to knowledge, call done.

    Asserts the verbatim system prompt drives the loop, the model's
    ``collection_write`` tool call lands a summary in the ``knowledge``
    memory attributed to the extractor identity, and the cursor advances
    so the next run sees no new pages.
    """
    config = make_config(history_interval=99999.0)
    requests_seen: list[dict] = []
    summary = (
        "https://tubesteader.com/products/eggnog\n\n"
        "The TubeSteader Eggnog is a single-channel tube overdrive pedal driven by "
        "a 12AX7 vacuum tube at 250 VDC, designed for boutique guitar rigs."
    )

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_llm._make_tool_call_response(
                request, "log_read_next", {"memory": "browse-results"}
            )
        if count == 2:
            return mock_llm._make_tool_call_response(
                request,
                "collection_get",
                {"memory": "knowledge", "key": "TubeSteader Eggnog"},
            )
        if count == 3:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "knowledge",
                    "entries": [{"key": "TubeSteader Eggnog", "content": summary}],
                },
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_browse_page(
            penny,
            "https://tubesteader.com/products/eggnog",
            "TubeSteader Eggnog",
            "The Eggnog uses a 12AX7 tube driven at 250 VDC.",
        )

        result = await penny.knowledge_extractor_agent.execute()

        assert result is True

        # Full exact system prompt the model saw on its first step
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

You extract durable knowledge from web pages Penny has read.

1. Call log_read_next("browse-results") to fetch new browse \
entries.  Each entry is one page (URL line, Title line, then \
page content).
2. For each page entry, write a single dense paragraph of 8-12 \
sentences capturing the key factual content.  Focus on:
   - What the thing IS (product, article, concept, etc.)
   - Specific details that would be useful to recall later \
(specs, names, dates, claims, findings)
   - What makes it notable or distinctive
   Do NOT include navigation/ads/site chrome, \
"This page describes..." meta-framing, opinions about content \
quality, or anything not on the page.  Plain declarative \
prose; no bullets, no markdown, no headers.
3. For each page, call collection_get("knowledge", key=<page \
title>) to see whether you already have a summary.  If one is \
returned, call collection_update("knowledge", key=<title>, \
content=<merged paragraph>) — integrate any new details from \
this fetch while preserving existing ones.  Otherwise, call \
collection_write("knowledge", entries=[{key: <title>, \
content: <new paragraph>}]).
4. Call done().

The entry's content should start with the page URL on its own \
line, then a blank line, then the summary paragraph — so \
retrieval can render the source link alongside the summary.

If no new browse entries appear, call done() without writing \
anything."""
        assert rest == expected, (
            f"Knowledge extractor system prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"
        )

        # Summary landed in the knowledge collection, attributed to the extractor
        knowledge = penny.db.memories.read_all("knowledge")
        eggnog = next((e for e in knowledge if e.key == "TubeSteader Eggnog"), None)
        assert eggnog is not None
        assert eggnog.author == KnowledgeExtractorAgent.name
        assert "12AX7" in eggnog.content
        assert eggnog.content.startswith("https://tubesteader.com/products/eggnog")

        # Cursor advanced past the seeded page
        cursor = _cursor_for_knowledge_extractor(penny)
        assert cursor is not None


@pytest.mark.asyncio
async def test_knowledge_cursor_does_not_advance_on_max_steps(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Failed run leaves the knowledge cursor untouched so the next pass replays."""
    config = make_config(history_interval=99999.0)

    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_tool_call_response(
            request, "log_read_next", {"memory": "browse-results"}
        )
    )

    async with running_penny(config) as penny:
        _seed_browse_page(penny, "https://example.com", "Example", "Some content.")
        before = _cursor_for_knowledge_extractor(penny)

        result = await penny.knowledge_extractor_agent.execute()

        assert result is False
        after = _cursor_for_knowledge_extractor(penny)
        assert after == before


@pytest.mark.asyncio
async def test_no_browse_pages_completes_cleanly(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Empty browse-results log → model reads nothing, calls done immediately."""
    config = make_config(history_interval=99999.0)

    def handler(request, _count):
        messages = request.get("messages", [])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            return mock_llm._make_tool_call_response(
                request, "log_read_next", {"memory": "browse-results"}
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Seed only the entries the migration backfilled — no fresh pages.
        before = penny.db.memories.read_all("knowledge")

        result = await penny.knowledge_extractor_agent.execute()

        assert result is True
        # No new entries written
        after = penny.db.memories.read_all("knowledge")
        assert len(after) == len(before)


def test_knowledge_extractor_max_steps_constant_is_set():
    """Defensive check: the cap is non-zero so the loop can actually run."""
    assert KnowledgeExtractorAgent.MAX_STEPS > 0
