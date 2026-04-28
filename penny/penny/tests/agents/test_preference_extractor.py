"""Integration tests for PreferenceExtractorAgent.

Per-user agent that reads new entries from ``user-messages`` via
``log_read_next``, identifies likes / dislikes, and writes them to
the ``likes`` / ``dislikes`` collections.  Cursor commits only on a
clean ``done()`` exit.

Test organisation:
1. Happy path — verbatim system prompt + cursor advancement on success
2. Failure — cursor stays put on max_steps
3. Empty log — done immediately, no writes
"""

from __future__ import annotations

import pytest

from penny.agents.preference_extractor import PreferenceExtractorAgent
from penny.constants import PennyConstants
from penny.database.memory_store import LogEntryInput


def _seed_user_message(penny, content: str) -> None:
    """Seed an entry into the user-messages log (simulates channel ingress)."""
    penny.db.memories.append(
        PennyConstants.MEMORY_USER_MESSAGES_LOG,
        [LogEntryInput(content=content)],
        author="user",
    )


def _cursor(penny):
    return penny.db.cursors.get(
        PreferenceExtractorAgent.name, PennyConstants.MEMORY_USER_MESSAGES_LOG
    )


# ── 1. Happy path ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preference_extraction_full_loop(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Core success flow: read user-messages, write to likes, call done.

    Asserts the verbatim system prompt drives the loop, the model's
    ``collection_write`` tool call lands an entry in the ``likes`` memory,
    and the cursor advances so the next run sees no new messages.
    """
    config = make_config(preference_extractor_interval=99999.0)
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

        result = await penny.preference_extractor_agent.execute()

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
        assert _cursor(penny) is not None


# ── 2. Failure: cursor stays on max_steps ────────────────────────────────


@pytest.mark.asyncio
async def test_cursor_does_not_advance_on_max_steps(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """If the model exhausts max_steps without calling done, the cursor stays
    where it was so the next run sees the same messages again."""
    config = make_config(preference_extractor_interval=99999.0)

    # Always return log_read_next — never done.  Loop hits max_steps.
    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_tool_call_response(
            request, "log_read_next", {"memory": "user-messages"}
        )
    )

    async with running_penny(config) as penny:
        _seed_user_message(penny, "first message")
        before = _cursor(penny)

        result = await penny.preference_extractor_agent.execute()

        assert result is False
        assert _cursor(penny) == before


# ── 3. Empty log ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_user_messages_completes_cleanly(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Empty user-messages log → model reads nothing, calls done immediately."""
    config = make_config(preference_extractor_interval=99999.0)

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
        result = await penny.preference_extractor_agent.execute()

        assert result is True
        assert penny.db.memories.read_all("likes") == []
        assert penny.db.memories.read_all("dislikes") == []
