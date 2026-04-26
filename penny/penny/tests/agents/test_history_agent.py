"""Integration tests for HistoryAgent: preference extraction and knowledge extraction.

Test organisation:
1. Preference extraction (agent shell) — happy path with full verbatim system
   prompt + cursor advancement on success / no advancement on failure
2. Knowledge extraction — bespoke flow, unchanged from before the memory port
"""

from __future__ import annotations

import json

import pytest

from penny.agents.history import (
    PREFERENCE_EXTRACTOR_AGENT_NAME,
    PREFERENCE_EXTRACTOR_MAX_STEPS,
    HistoryAgent,
)
from penny.constants import PennyConstants
from penny.database.memory_store import LogEntryInput
from penny.database.models import PromptLog
from penny.tests.conftest import TEST_SENDER


def _seed_user_message(penny, content: str) -> None:
    """Seed an entry into the user-messages log (simulates channel ingress)."""
    penny.db.memories.append(
        PennyConstants.MEMORY_USER_MESSAGES_LOG,
        [LogEntryInput(content=content)],
        author="user",
    )


def _cursor_for_preference_extractor(penny):
    return penny.db.cursors.get(
        PREFERENCE_EXTRACTOR_AGENT_NAME, PennyConstants.MEMORY_USER_MESSAGES_LOG
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

        result = await penny.history_agent.execute_for_user(TEST_SENDER)

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
        assert coffee.author == PREFERENCE_EXTRACTOR_AGENT_NAME

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

        result = await penny.history_agent.execute_for_user(TEST_SENDER)

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
        result = await penny.history_agent.execute_for_user(TEST_SENDER)

        assert result is True
        assert penny.db.memories.read_all("likes") == []
        assert penny.db.memories.read_all("dislikes") == []


def test_preference_extractor_max_steps_constant_is_set():
    """Defensive check: the cap is non-zero so the loop can actually run."""
    assert PREFERENCE_EXTRACTOR_MAX_STEPS > 0
    # Sanity: HistoryAgent.get_max_steps should reflect the constant.
    # We can't instantiate Agent without lots of config, so just check the
    # class-attribute path matches.
    assert HistoryAgent.get_max_steps.__qualname__ == "HistoryAgent.get_max_steps"


# ── Knowledge extraction ────────────────────────────────────────────────


def _insert_prompt_with_browse(penny, url, title, page_content):
    """Insert a prompt log with a browse tool result."""
    browse_header = PennyConstants.BROWSE_PAGE_HEADER
    tool_content = f"{browse_header}{url}\nTitle: {title}\nURL: {url}\n\n{page_content}"
    messages = json.dumps(
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        ]
    )
    with penny.db.get_session() as session:
        prompt = PromptLog(
            model="test",
            messages=messages,
            response=json.dumps({"choices": []}),
            agent_name="chat",
            prompt_type="user_message",
        )
        session.add(prompt)
        session.commit()
        session.refresh(prompt)
        return prompt.id


@pytest.mark.asyncio
async def test_extract_knowledge_from_browse_results(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge extraction creates an entry from browse tool results."""
    config = make_config(history_interval=99999.0)

    def handler(request, count):
        return mock_llm._make_text_response(
            request, "The Eggnog is a tube overdrive pedal by TubeSteader."
        )

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _insert_prompt_with_browse(
            penny,
            "https://tubesteader.com/products/eggnog",
            "TubeSteader Eggnog",
            "The Eggnog uses a 12AX7 tube driven at 250 VDC.",
        )

        await penny.history_agent._extract_knowledge()

        entry = penny.db.knowledge.get_by_url("https://tubesteader.com/products/eggnog")
        assert entry is not None
        assert entry.title == "TubeSteader Eggnog"
        assert "tube overdrive" in entry.summary


@pytest.mark.asyncio
async def test_extract_knowledge_upserts_existing_url(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Re-browsing the same URL aggregates into the existing knowledge entry."""
    config = make_config(history_interval=99999.0)

    call_count = 0

    def handler(request, count):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_llm._make_text_response(request, "First summary of the page.")
        return mock_llm._make_text_response(request, "Updated summary with new info.")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _insert_prompt_with_browse(
            penny,
            "https://example.com/page",
            "Example Page",
            "Original content.",
        )
        await penny.history_agent._extract_knowledge()

        entry = penny.db.knowledge.get_by_url("https://example.com/page")
        assert entry is not None
        assert entry.summary == "First summary of the page."

        # Insert another prompt with the same URL
        _insert_prompt_with_browse(
            penny,
            "https://example.com/page",
            "Example Page",
            "Updated content with more details.",
        )
        await penny.history_agent._extract_knowledge()

        entry = penny.db.knowledge.get_by_url("https://example.com/page")
        assert entry is not None
        assert entry.summary == "Updated summary with new info."


@pytest.mark.asyncio
async def test_extract_knowledge_dedupes_same_url_within_batch(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Same URL re-logged across an agentic loop's steps is summarized once.

    Each step of an agentic loop re-logs the prior tool result messages, so a
    single browse appears in multiple PromptLog rows. Knowledge extraction must
    process that URL exactly once per batch (latest content wins) instead of
    re-aggregating identical content for every step.
    """
    config = make_config(history_interval=99999.0)

    summaries_generated = []

    def handler(request, count):
        summary = f"Summary {len(summaries_generated) + 1}"
        summaries_generated.append(summary)
        return mock_llm._make_text_response(request, summary)

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Simulate three steps of an agentic loop, each re-logging the same
        # browse tool result. The third step has the freshest content.
        _insert_prompt_with_browse(
            penny, "https://loop.example/page", "Loop Page", "Step 1 content"
        )
        _insert_prompt_with_browse(
            penny, "https://loop.example/page", "Loop Page", "Step 2 content"
        )
        _insert_prompt_with_browse(
            penny, "https://loop.example/page", "Loop Page", "Step 3 content"
        )

        await penny.history_agent._extract_knowledge()

        # Exactly one LLM call (the dedup), not three.
        assert len(summaries_generated) == 1
        entry = penny.db.knowledge.get_by_url("https://loop.example/page")
        assert entry is not None
        assert entry.summary == "Summary 1"


@pytest.mark.asyncio
async def test_extract_knowledge_dedupes_url_fragments_within_batch(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """`/page` and `/page#anchor` are the same page; collapse to one summarize.

    The browser tool sometimes follows in-page anchor links from search
    results, producing browse entries whose URLs differ only by `#fragment`.
    Without normalization the dedup keys them as distinct URLs and the model
    summarizes byte-identical content N times, also writing N rows to the
    knowledge table that all point at the same page.
    """
    config = make_config(history_interval=99999.0)

    summaries_generated = []

    def handler(request, count):
        summary = f"Summary {len(summaries_generated) + 1}"
        summaries_generated.append(summary)
        return mock_llm._make_text_response(request, summary)

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _insert_prompt_with_browse(penny, "https://Example.com/page", "Example", "Same content")
        _insert_prompt_with_browse(
            penny, "https://example.com/page#intro", "Example", "Same content"
        )
        _insert_prompt_with_browse(
            penny, "https://example.com/page#summary", "Example", "Same content"
        )

        await penny.history_agent._extract_knowledge()

        # Exactly one LLM call across the three fragment variants.
        assert len(summaries_generated) == 1

        # Knowledge is stored under the canonical (fragment-stripped, lowercase
        # host) URL and no duplicate rows exist for the fragment variants.
        entry = penny.db.knowledge.get_by_url("https://example.com/page")
        assert entry is not None
        assert penny.db.knowledge.get_by_url("https://example.com/page#intro") is None
        assert penny.db.knowledge.get_by_url("https://example.com/page#summary") is None
        assert penny.db.knowledge.get_by_url("https://Example.com/page") is None


def test_normalize_url_strips_fragment_and_lowercases_host():
    """`_normalize_url` canonicalizes URLs for dedup keying."""
    normalize = HistoryAgent._normalize_url
    # Fragment stripped
    assert normalize("https://example.com/page#anchor") == "https://example.com/page"
    # Host lowercased, scheme lowercased
    assert normalize("HTTPS://Example.COM/Path") == "https://example.com/Path"
    # Path case preserved (servers can be case-sensitive)
    assert (
        normalize("https://example.com/CaseSensitive/Path#x")
        == "https://example.com/CaseSensitive/Path"
    )
    # Query string preserved
    assert (
        normalize("https://example.com/search?q=Foo&p=1#results")
        == "https://example.com/search?q=Foo&p=1"
    )
    # Already-canonical URL is unchanged
    assert normalize("https://example.com/page") == "https://example.com/page"


@pytest.mark.asyncio
async def test_extract_knowledge_respects_watermark(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge extraction only processes prompts after the watermark."""
    config = make_config(history_interval=99999.0)

    summaries_generated = []

    def handler(request, count):
        summary = f"Summary {len(summaries_generated) + 1}"
        summaries_generated.append(summary)
        return mock_llm._make_text_response(request, summary)

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        _insert_prompt_with_browse(penny, "https://a.com", "Page A", "Content A")
        _insert_prompt_with_browse(penny, "https://b.com", "Page B", "Content B")

        # First extraction processes both
        await penny.history_agent._extract_knowledge()
        assert penny.db.knowledge.get_by_url("https://a.com") is not None
        assert penny.db.knowledge.get_by_url("https://b.com") is not None

        summaries_generated.clear()

        # Add a third prompt
        _insert_prompt_with_browse(penny, "https://c.com", "Page C", "Content C")
        await penny.history_agent._extract_knowledge()

        # Only the new prompt should be processed
        assert len(summaries_generated) == 1
        assert penny.db.knowledge.get_by_url("https://c.com") is not None


def _insert_prompt_without_browse(penny):
    """Insert a prompt log with no browse tool results."""
    messages = json.dumps(
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hey penny what's up"},
            {"role": "assistant", "content": "not much!"},
        ]
    )
    with penny.db.get_session() as session:
        prompt = PromptLog(
            model="test",
            messages=messages,
            response=json.dumps({"choices": []}),
            agent_name="chat",
            prompt_type="user_message",
        )
        session.add(prompt)
        session.commit()
        session.refresh(prompt)
        return prompt.id


@pytest.mark.asyncio
async def test_extract_knowledge_skips_prompts_without_browse(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Knowledge extraction only queries prompts containing browse results."""
    config = make_config(history_interval=99999.0)

    summaries_generated = []

    def handler(request, count):
        summary = f"Summary {len(summaries_generated) + 1}"
        summaries_generated.append(summary)
        return mock_llm._make_text_response(request, summary)

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Insert many prompts without browse results
        for _ in range(10):
            _insert_prompt_without_browse(penny)

        # Insert one prompt with browse results
        _insert_prompt_with_browse(penny, "https://found.com", "Found Page", "Content")

        await penny.history_agent._extract_knowledge()

        # Only the browse-containing prompt should be processed
        assert len(summaries_generated) == 1
        assert penny.db.knowledge.get_by_url("https://found.com") is not None


# ── Browse section parsing (unit tests) ─────────────────────────────────


_HEADER = PennyConstants.BROWSE_PAGE_HEADER


def test_parse_browse_section_healthy():
    """Healthy browse result with Title + URL + content is parsed."""
    section = (
        f"{_HEADER}https://example.com/page\n"
        "Title: Example Page\n"
        "URL: https://example.com/page\n"
        "\nThis is the page content with lots of useful information."
    )
    result = HistoryAgent._parse_browse_section(section)
    assert result is not None
    url, title, content = result
    assert url == "https://example.com/page"
    assert title == "Example Page"
    assert "useful information" in content


def test_parse_browse_section_rejects_error_section_header():
    """Sections under the dedicated error header never reach the parser, but if
    one were passed in directly it would be rejected (no Title:/URL: lines)."""
    section = (
        f"{PennyConstants.BROWSE_ERROR_HEADER}https://example.com\n"
        "Could not read page: extraction failed after 10 retries"
    )
    assert HistoryAgent._parse_browse_section(section) is None


def test_parse_browse_section_missing_title_line():
    """Sections without a Title: line are rejected."""
    section = f"{_HEADER}https://example.com\nNo title here\nURL: https://example.com\nbody"
    assert HistoryAgent._parse_browse_section(section) is None


def test_parse_browse_section_missing_url_line():
    """Sections without a URL: line are rejected."""
    section = f"{_HEADER}https://example.com\nTitle: Some Page\nNo url here\nbody"
    assert HistoryAgent._parse_browse_section(section) is None


def test_parse_browse_section_empty_body():
    """Page with Title + URL but empty body is rejected — nothing to summarize."""
    section = f"{_HEADER}https://example.com\nTitle: Some Page\nURL: https://example.com\n"
    assert HistoryAgent._parse_browse_section(section) is None


def test_parse_browse_section_whitespace_body():
    """Page with Title + URL but whitespace-only body is rejected."""
    section = f"{_HEADER}https://example.com\nTitle: Some Page\nURL: https://example.com\n   \n\n"
    assert HistoryAgent._parse_browse_section(section) is None


def test_parse_browse_section_single_line():
    """Single-line browse header with no content is rejected."""
    section = f"{_HEADER}https://example.com"
    assert HistoryAgent._parse_browse_section(section) is None


def test_parse_browse_results_skips_error_sections():
    """Mixed tool message: a healthy section is parsed, an error section is skipped."""
    from penny.database.models import PromptLog

    healthy = (
        f"{PennyConstants.BROWSE_PAGE_HEADER}https://good.com\n"
        "Title: Good Page\n"
        "URL: https://good.com\n"
        "\nReal page content with substance."
    )
    error = (
        f"{PennyConstants.BROWSE_ERROR_HEADER}https://bad.com\n"
        "Could not read page: extraction failed after 10 retries"
    )
    tool_content = PennyConstants.SECTION_SEPARATOR.join([healthy, error])
    prompt = PromptLog(
        model="test",
        messages=json.dumps(
            [
                {"role": "user", "content": "test"},
                {"role": "tool", "tool_call_id": "x", "content": tool_content},
            ]
        ),
        response="{}",
    )
    results = HistoryAgent._parse_browse_results(prompt)
    assert len(results) == 1
    assert results[0][0] == "https://good.com"
    assert "substance" in results[0][2]
