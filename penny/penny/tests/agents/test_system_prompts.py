"""System prompt structural-drift tests.

Asserts on the exact ``##``/``###`` header sequence each agent's
``_build_system_prompt`` produces under a deterministic baseline state
(profile only — no thoughts, knowledge, preferences, or related messages).

If a building block is added, removed, or reordered, these tests fail
loudly.  Update the expected sequence here when the change is intentional.
"""

from __future__ import annotations

import re

import pytest

from penny.agents.notify import CheckinMode, ThoughtMode
from penny.tests.conftest import TEST_SENDER

_HEADER_RE = re.compile(r"^(#+)\s+(.+)$")


def _extract_headers(prompt: str) -> list[tuple[int, str]]:
    """Return ``(level, title)`` tuples for every markdown header in order."""
    return [
        (len(match.group(1)), match.group(2))
        for line in prompt.split("\n")
        if (match := _HEADER_RE.match(line))
    ]


@pytest.mark.asyncio
async def test_chat_agent_system_prompt_structure(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """ChatAgent: identity → context (profile) → instructions, no extras."""
    async with running_penny(test_config) as penny:
        penny.chat_agent._pending_page_context = None
        prompt = await penny.chat_agent._build_system_prompt(TEST_SENDER)

    assert _extract_headers(prompt) == [
        (2, "Identity"),
        (2, "Context"),
        (3, "User Profile"),
        (2, "Instructions"),
    ]


@pytest.mark.asyncio
async def test_notify_checkin_system_prompt_structure(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """CheckinMode: identity → context (profile, no notified thoughts) → instructions."""
    async with running_penny(test_config) as penny:
        mode = CheckinMode()
        prompt = mode.build_system_prompt(penny.notify_agent, TEST_SENDER)

    assert _extract_headers(prompt) == [
        (2, "Identity"),
        (2, "Context"),
        (3, "User Profile"),
        (2, "Instructions"),
    ]


@pytest.mark.asyncio
async def test_notify_thought_system_prompt_structure(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """ThoughtMode: identity → context (profile, no pending thought) → instructions."""
    async with running_penny(test_config) as penny:
        penny.notify_agent._pending_thought = None
        mode = ThoughtMode(thought=None, config=penny.config)
        prompt = mode.build_system_prompt(penny.notify_agent, TEST_SENDER)

    assert _extract_headers(prompt) == [
        (2, "Identity"),
        (2, "Context"),
        (3, "User Profile"),
        (2, "Instructions"),
    ]


@pytest.mark.asyncio
async def test_thinking_agent_system_prompt_structure(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """ThinkingAgent: no identity, no profile — just instructions when no seed/dislikes."""
    async with running_penny(test_config) as penny:
        penny.thinking_agent._seed_pref_id = None
        prompt = await penny.thinking_agent._build_system_prompt(TEST_SENDER)

    assert _extract_headers(prompt) == [(2, "Instructions")]
