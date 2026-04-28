"""Integration tests for NotifyAgent — fully model-driven shell.

The agent is just ``name`` + ``terminator_tool``;
the prompt steers the model through reading ``unnotified-thoughts``,
picking one, moving it to ``notified-thoughts``, and dispatching via
``send_message``.  Mute and cooldown gating live inside
``SendMessageTool``; if no unnotified thoughts exist, the prompt
tells the model to call ``done()``.

Test organisation:
1. Happy path — full verbatim system prompt + model-driven sequence
2. Skip cases — model calls ``done()`` without sending
"""

from __future__ import annotations

import pytest

from penny.database.memory_store import EntryInput
from penny.tests.conftest import wait_until


def _seed_unnotified_thought(penny, key: str, content: str) -> None:
    """Seed an entry into the unnotified-thoughts collection."""
    penny.db.memories.write(
        "unnotified-thoughts",
        [EntryInput(key=key, content=content)],
        author="thinking",
    )


# ── 1. Happy path ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_cycle_happy_path(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Core success flow: read unnotified, move it, send_message, done.

    Asserts the verbatim system prompt drives the loop, the model's
    ``collection_move`` tool call lands the entry in ``notified-thoughts``,
    and the model's ``send_message`` tool call delivers through the channel.
    """
    config = make_config()
    requests_seen: list[dict] = []
    notification_text = (
        "hey! been chewing on this Tubesteader Beekeeper review — clean fuzz, "
        "low-noise floor, sounds gnarly 🐝"
    )

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_llm._make_tool_call_response(
                request,
                "read_latest",
                {"memory": "unnotified-thoughts"},
            )
        if count == 2:
            return mock_llm._make_tool_call_response(
                request,
                "log_read_recent",
                {"memory": "penny-messages", "window_seconds": 86400},
            )
        if count == 3:
            return mock_llm._make_tool_call_response(
                request,
                "collection_move",
                {
                    "key": "Tubesteader Beekeeper review",
                    "from_memory": "unnotified-thoughts",
                    "to_memory": "notified-thoughts",
                },
            )
        if count == 4:
            return mock_llm._make_tool_call_response(
                request, "send_message", {"content": notification_text}
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_unnotified_thought(
            penny,
            "Tubesteader Beekeeper review",
            "Found a great review of the Tubesteader Beekeeper fuzz pedal 🐝",
        )

        result = await penny.notify_agent.execute()

        assert result is True

        # Full exact system prompt the model saw on its first step
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

You are Penny's notify agent. Once per cycle, you reach out to \
your friend the user with ONE thought worth sharing.

Sequence:
1. read_latest("unnotified-thoughts") — list every \
fresh thought you have to share.
2. log_read_recent("penny-messages", window_seconds=86400) — \
see what you've already said today; don't repeat yourself.
3. Pick ONE unnotified thought you haven't already shared and \
still find interesting.
4. collection_move("unnotified-thoughts", "notified-thoughts", \
key=<chosen key>) — mark it as shared.
5. send_message(content=<your message>) — deliver the thought to \
the user.  Write it conversationally, like you're texting a \
friend; open with a casual greeting, then write out every \
detail in full.  No ellipses ('...', '…'), no 'etc.', no 'and \
more', no teaser phrasing — finish every sentence and bullet \
you start.  The user only sees what you put in `content`; \
nothing else is attached.  Include the specific details from \
the thought (names, specs, dates), at least one source URL \
from the thought, and finish with an emoji.
6. done().

Every fact and URL in your message must come from the thought \
you read — do not invent information.  If no unnotified thought \
is worth sharing, call done() without sending anything."""
        assert rest == expected, (
            f"Notify system prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"
        )

        # Message landed via Signal
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)
        sent = signal_server.outgoing_messages[-1]["message"]
        assert "Tubesteader Beekeeper" in sent

        # Thought moved between collections
        unnotified_keys = [e.key for e in penny.db.memories.read_all("unnotified-thoughts")]
        notified_keys = [e.key for e in penny.db.memories.read_all("notified-thoughts")]
        assert "Tubesteader Beekeeper review" not in unnotified_keys
        assert "Tubesteader Beekeeper review" in notified_keys


# ── 2. Skip cases ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_message_sent_when_model_calls_done(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Model exits via ``done()`` without sending — cycle reports False, nothing sent."""
    config = make_config()

    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_tool_call_response(request, "done", {})
    )

    async with running_penny(config) as penny:
        _seed_unnotified_thought(penny, "Boring topic", "nothing worth sharing")

        result = await penny.notify_agent.execute()

        assert result is False
        assert signal_server.outgoing_messages == []
