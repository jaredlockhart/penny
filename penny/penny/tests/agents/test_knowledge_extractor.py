"""Integration tests for KnowledgeExtractorAgent.

User-independent agent that reads new entries from ``browse-results``
via ``log_read_next``, summarizes each page, and writes summaries to
the ``knowledge`` collection.  ``execute()`` skips user iteration
entirely.

Test organisation:
1. Happy path — verbatim system prompt + cursor advancement on success
2. Failure — cursor stays put on max_steps
3. Empty log — done immediately, no writes
"""

from __future__ import annotations

import pytest

from penny.agents.knowledge_extractor import KnowledgeExtractorAgent
from penny.constants import PennyConstants
from penny.database.memory_store import LogEntryInput


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


def _cursor(penny):
    return penny.db.cursors.get(
        KnowledgeExtractorAgent.name, PennyConstants.MEMORY_BROWSE_RESULTS_LOG
    )


# ── 1. Happy path ────────────────────────────────────────────────────────


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
    config = make_config(knowledge_extractor_interval=99999.0)
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

### Memory Inventory
- browse-results (log) — Every browse-tool fetch result
- dislikes (collection) — Topics the user has expressed negative sentiment about
- knowledge (collection) — Summarized facts from web pages Penny has read
- likes (collection) — Topics the user has expressed positive sentiment about
- notified-thoughts (collection) — Thoughts already shared with the user
- penny-messages (log) — Every outgoing Penny reply
- unnotified-thoughts (collection) — Pending thoughts to share with the user
- user-messages (log) — Every incoming user message

## Instructions
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
        assert _cursor(penny) is not None


# ── 2. Failure: cursor stays on max_steps ────────────────────────────────


@pytest.mark.asyncio
async def test_cursor_does_not_advance_on_max_steps(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Failed run leaves the knowledge cursor untouched so the next pass replays."""
    config = make_config(knowledge_extractor_interval=99999.0)

    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_tool_call_response(
            request, "log_read_next", {"memory": "browse-results"}
        )
    )

    async with running_penny(config) as penny:
        _seed_browse_page(penny, "https://example.com", "Example", "Some content.")
        before = _cursor(penny)

        result = await penny.knowledge_extractor_agent.execute()

        assert result is False
        assert _cursor(penny) == before


# ── 3. Empty log ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_browse_pages_completes_cleanly(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Empty browse-results log → model reads nothing, calls done immediately."""
    config = make_config(knowledge_extractor_interval=99999.0)

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
        before = penny.db.memories.read_all("knowledge")

        result = await penny.knowledge_extractor_agent.execute()

        assert result is True
        # No new entries written
        after = penny.db.memories.read_all("knowledge")
        assert len(after) == len(before)
