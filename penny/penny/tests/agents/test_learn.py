"""Tests for LearnAgent pipeline gating."""

import json

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_learn_gated_by_unextracted_search_logs(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """LearnAgent skips when unextracted learn_command search logs exist."""
    config = make_config()
    mock_ollama.set_response_handler(lambda req, count: mock_ollama._make_text_response(req, "ok"))

    async with running_penny(config) as penny:
        # Create an active learn prompt
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER, prompt_text="kef speakers", searches_remaining=3
        )
        assert lp is not None and lp.id is not None

        # Create an unextracted learn_command search log
        penny.db.searches.log(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger=PennyConstants.SearchTrigger.LEARN_COMMAND,
            learn_prompt_id=lp.id,
        )

        agent = penny.learn_agent

        # Gate should be closed — agent should skip
        result = await agent.execute()
        assert result is False, "LearnAgent should skip when unextracted learn search logs exist"

        # Still only 1 search log (no new search created)
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        assert len(search_logs) == 1


@pytest.mark.asyncio
async def test_learn_proceeds_when_search_logs_extracted(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """LearnAgent proceeds after search logs are extracted (gate opens)."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""
        if "search query" in last_content.lower():
            return mock_ollama._make_text_response(
                request, json.dumps({"query": "kef speakers review"})
            )
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create an active learn prompt with an extracted search log
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER, prompt_text="kef speakers", searches_remaining=2
        )
        assert lp is not None and lp.id is not None

        penny.db.searches.log(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger=PennyConstants.SearchTrigger.LEARN_COMMAND,
            learn_prompt_id=lp.id,
        )
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        penny.db.searches.mark_extracted(search_logs[0].id)

        agent = penny.learn_agent

        # Gate is open — agent should execute a new search
        result = await agent.execute()
        assert result is True, "LearnAgent should proceed when all learn search logs are extracted"

        # Now 2 search logs (original + new)
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        assert len(search_logs) == 2


@pytest.mark.asyncio
async def test_learn_not_gated_by_user_message_search_logs(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """Unextracted user_message search logs do NOT gate the LearnAgent."""
    config = make_config()

    def handler(request: dict, count: int) -> dict:
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""
        if "search query" in last_content.lower():
            return mock_ollama._make_text_response(
                request, json.dumps({"query": "kef speakers review"})
            )
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(config) as penny:
        # Create an unextracted user_message search log
        penny.db.messages.log_message(direction="incoming", sender=TEST_SENDER, content="hello")
        penny.db.searches.log(
            query="hello",
            response="Some results...",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )

        # Create an active learn prompt (no learn search logs yet)
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER, prompt_text="kef speakers", searches_remaining=3
        )
        assert lp is not None

        agent = penny.learn_agent

        # Gate should NOT be closed — user_message logs don't gate learn
        result = await agent.execute()
        assert result is True, "LearnAgent should not be gated by user_message search logs"


@pytest.mark.asyncio
async def test_has_unextracted_learn_search_logs_db_method(
    signal_server,
    mock_ollama,
    _mock_search,
    make_config,
    test_user_info,
    running_penny,
):
    """DB method returns True only for unextracted learn_command search logs."""
    config = make_config()

    async with running_penny(config) as penny:
        # No search logs at all
        assert penny.db.searches.has_unextracted_learn_logs() is False

        # Add unextracted user_message search log — should still be False
        penny.db.searches.log(
            query="test",
            response="result",
            trigger=PennyConstants.SearchTrigger.USER_MESSAGE,
        )
        assert penny.db.searches.has_unextracted_learn_logs() is False

        # Add unextracted learn_command search log — now True
        lp = penny.db.learn_prompts.create(
            user=TEST_SENDER, prompt_text="test topic", searches_remaining=3
        )
        assert lp is not None and lp.id is not None
        penny.db.searches.log(
            query="test topic",
            response="result",
            trigger=PennyConstants.SearchTrigger.LEARN_COMMAND,
            learn_prompt_id=lp.id,
        )
        assert penny.db.searches.has_unextracted_learn_logs() is True

        # Mark it extracted — back to False
        search_logs = penny.db.searches.get_by_learn_prompt(lp.id)
        penny.db.searches.mark_extracted(search_logs[0].id)
        assert penny.db.searches.has_unextracted_learn_logs() is False
