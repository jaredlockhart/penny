"""Integration tests for ThinkingAgent — fully model-driven shell.

The agent has no bespoke logic anymore: a single shell calls the agent
loop with the full tool surface, and the prompt steers everything.

Test organisation:
1. Happy path — full verbatim system prompt + done() exits with True
2. Failure case — model never calls done() → returns False
"""

from __future__ import annotations

import pytest

from penny.agents.preference_extractor import PreferenceExtractorAgent
from penny.database.memory_store import EntryInput


def _seed_likes(penny, *topics: str) -> None:
    for topic in topics:
        penny.db.memories.write(
            "likes",
            [EntryInput(key=topic, content=topic)],
            author=PreferenceExtractorAgent.name,
        )


# ── 1. Happy path ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thinking_cycle_happy_path(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Core success flow: read seed, browse, write a thought, call done.

    Asserts the verbatim system prompt drives the loop and the model's
    write to ``unnotified-thoughts`` lands as expected.
    """
    config = make_config(max_steps=8)
    requests_seen: list[dict] = []

    def handler(request, count):
        requests_seen.append(request)
        if count == 1:
            return mock_llm._make_tool_call_response(
                request,
                "collection_read_random",
                {"memory": "likes", "k": 1},
            )
        if count == 2:
            return mock_llm._make_tool_call_response(request, "read_all", {"memory": "dislikes"})
        if count == 3:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "unnotified-thoughts",
                    "entries": [
                        {
                            "key": "Tubesteader Beekeeper review",
                            "content": "found a great review of the Beekeeper pedal 🐝",
                        }
                    ],
                },
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _seed_likes(penny, "guitar pedals")

        result = await penny.thinking_agent.execute()

        assert result is True

        # Full exact system prompt the model saw on its first step
        system_text = [
            m.get("content", "") for m in requests_seen[0]["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        expected = """\

You are Penny's thinking agent. Once per run, you find ONE specific, \
concrete thing worth knowing about — something the user would enjoy \
hearing — and store it as a thought.

Sequence:
1. collection_read_random("likes", 1) — pick one seed topic from the \
user's likes.
2. read_all("dislikes") — see what the user doesn't like.
3. browse — search the web and read one or two pages to find something \
timely and interesting grounded in the seed topic.
4. Draft ONE thought connecting what you found to the seed.  Write it \
conversationally, like you're texting a friend; include specific \
details (names, specs, dates), at least one source URL, and finish \
with an emoji.  Keep it under 300 words.
5. Check the draft against the dislikes list.  If it conflicts with \
anything the user dislikes, call done() without writing.
6. exists(["unnotified-thoughts", "notified-thoughts"], key, content) \
— if a similar thought already exists, call done() without writing.
7. collection_write("unnotified-thoughts", entries=[{key: short topic \
name (3-10 words), content: the thought you drafted}]).
8. done().

The interesting stuff is ON the pages, not in search snippets — \
browse the URLs you find rather than searching forever.  If nothing \
noteworthy comes up, call done() without writing; quiet cycles are \
normal.  Troubleshooting guides, bug workarounds, and support \
articles are NOT interesting discoveries."""
        assert rest == expected, (
            f"Thinking system prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"
        )

        # Write landed in the unnotified-thoughts collection
        thoughts = penny.db.memories.read_all("unnotified-thoughts")
        assert any(e.key == "Tubesteader Beekeeper review" for e in thoughts)
        new_entry = next(e for e in thoughts if e.key == "Tubesteader Beekeeper review")
        assert new_entry.author == "thinking"


# ── 2. Failure: model exhausts steps without calling done ─────────────────


@pytest.mark.asyncio
async def test_returns_false_when_model_does_not_call_done(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """No done() call → run is treated as failed; result is False."""
    config = make_config(max_steps=2)

    # Always return read_all — never done.  Loop hits max_steps.
    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_tool_call_response(
            request, "read_all", {"memory": "likes"}
        )
    )

    async with running_penny(config) as penny:
        _seed_likes(penny, "jazz")
        result = await penny.thinking_agent.execute()
        assert result is False
