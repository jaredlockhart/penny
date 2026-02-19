"""Integration tests for /learn command."""

import json

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER, wait_until


@pytest.mark.asyncio
async def test_learn_iterative_search(
    signal_server, test_config, mock_ollama, _mock_search, running_penny
):
    """Search-based /learn creates LearnPrompt, iteratively generates and runs searches."""
    call_count = 0

    def handler(request: dict, count: int) -> dict:
        nonlocal call_count
        messages = request.get("messages", [])
        last_content = messages[-1].get("content", "") if messages else ""

        if "Generate a good search query" in last_content:
            # Initial query generation
            return mock_ollama._make_text_response(
                request,
                json.dumps({"query": "kef speakers overview"}),
            )
        if "Generate the next search query" in last_content:
            call_count += 1
            if call_count <= 2:
                # Followup queries informed by previous results
                return mock_ollama._make_text_response(
                    request,
                    json.dumps({"query": f"kef followup query {call_count}"}),
                )
            # Signal research is complete
            return mock_ollama._make_text_response(
                request,
                json.dumps({"query": ""}),
            )
        return mock_ollama._make_text_response(request, "ok")

    mock_ollama.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/learn kef speakers")
        response = await signal_server.wait_for_message(timeout=10.0)

        # Immediate acknowledgment
        assert "Okay" in response["message"]
        assert "kef speakers" in response["message"]

        # Wait for background searches to complete
        await wait_until(
            lambda: (
                len(penny.db.get_user_learn_prompts(TEST_SENDER)) > 0
                and penny.db.get_user_learn_prompts(TEST_SENDER)[0].status
                == PennyConstants.LearnPromptStatus.COMPLETED
            )
        )

        # Verify LearnPrompt was created and completed
        learn_prompts = penny.db.get_user_learn_prompts(TEST_SENDER)
        assert len(learn_prompts) == 1
        lp = learn_prompts[0]
        assert lp.prompt_text == "kef speakers"
        assert lp.status == PennyConstants.LearnPromptStatus.COMPLETED

        # Verify searches are linked to the LearnPrompt
        # 1 initial + 2 followups = 3 (LLM returned empty on 3rd followup)
        assert lp.id is not None
        search_logs = penny.db.get_search_logs_by_learn_prompt(lp.id)
        assert len(search_logs) == 3
        for sl in search_logs:
            assert sl.trigger == PennyConstants.SearchTrigger.LEARN_COMMAND
            assert sl.learn_prompt_id == lp.id


@pytest.mark.asyncio
async def test_learn_no_search_tool_creates_prompt_only(
    signal_server, mock_ollama, _mock_search, make_config, running_penny
):
    """Without Perplexity API key, creates LearnPrompt but no background searches."""
    config = make_config(perplexity_api_key=None)

    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/learn kef ls50")
        response = await signal_server.wait_for_message(timeout=10.0)

        assert "kef ls50" in response["message"]

        # LearnPrompt created but no searches
        learn_prompts = penny.db.get_user_learn_prompts(TEST_SENDER)
        assert len(learn_prompts) == 1
        assert learn_prompts[0].prompt_text == "kef ls50"


@pytest.mark.asyncio
async def test_learn_no_args_shows_status(signal_server, test_config, mock_ollama, running_penny):
    """Test /learn with no args shows provenance chain."""
    async with running_penny(test_config) as penny:
        # Create a completed LearnPrompt with linked data
        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="find me stuff about speakers",
            searches_remaining=0,
        )
        assert lp is not None and lp.id is not None
        penny.db.update_learn_prompt_status(lp.id, PennyConstants.LearnPromptStatus.COMPLETED)

        # Create a search log linked to the LearnPrompt
        penny.db.log_search(
            query="kef speakers overview",
            response="KEF makes great speakers...",
            trigger=PennyConstants.SearchTrigger.LEARN_COMMAND,
            learn_prompt_id=lp.id,
        )

        # Create an entity and a fact linked to that search log
        entity = penny.db.get_or_create_entity(TEST_SENDER, "kef ls50 meta")
        assert entity is not None and entity.id is not None

        # Get the search log to link the fact
        search_logs = penny.db.get_search_logs_by_learn_prompt(lp.id)
        assert len(search_logs) == 1
        sl_id = search_logs[0].id

        penny.db.add_fact(
            entity_id=entity.id,
            content="Costs $1,599 per pair",
            source_search_log_id=sl_id,
        )

        await signal_server.push_message(sender=TEST_SENDER, content="/learn")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Queued learning" in response["message"]
        assert "find me stuff about speakers" in response["message"]
        assert "\u2713" in response["message"]  # Completed indicator
        assert "kef ls50 meta" in response["message"]
        assert "1 fact" in response["message"]


@pytest.mark.asyncio
async def test_learn_no_args_empty(signal_server, test_config, mock_ollama, running_penny):
    """Test /learn with no args when no LearnPrompts exist."""
    async with running_penny(test_config) as _penny:
        await signal_server.push_message(sender=TEST_SENDER, content="/learn")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Nothing being actively researched" in response["message"]


@pytest.mark.asyncio
async def test_learn_status_shows_active(signal_server, test_config, mock_ollama, running_penny):
    """Test /learn status shows remaining search count for active LearnPrompts."""
    async with running_penny(test_config) as penny:
        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="ai conferences in europe",
            searches_remaining=3,
        )
        assert lp is not None

        await signal_server.push_message(sender=TEST_SENDER, content="/learn")
        response = await signal_server.wait_for_message(timeout=5.0)

        assert "Queued learning" in response["message"]
        assert "ai conferences in europe" in response["message"]
        assert "3 searches left" in response["message"]


@pytest.mark.asyncio
async def test_learn_prompt_crud(signal_server, test_config, mock_ollama, running_penny):
    """LearnPrompt CRUD operations work correctly."""
    async with running_penny(test_config) as penny:
        # Create
        lp = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="find me stuff about speakers",
            searches_remaining=3,
        )
        assert lp is not None
        assert lp.id is not None
        assert lp.status == "active"
        assert lp.searches_remaining == 3

        # Read
        fetched = penny.db.get_learn_prompt(lp.id)
        assert fetched is not None
        assert fetched.prompt_text == "find me stuff about speakers"

        # Update status
        penny.db.update_learn_prompt_status(lp.id, "completed")
        updated = penny.db.get_learn_prompt(lp.id)
        assert updated is not None
        assert updated.status == "completed"

        # Active list
        lp2 = penny.db.create_learn_prompt(
            user=TEST_SENDER,
            prompt_text="ai conferences in europe",
            searches_remaining=3,
        )
        active = penny.db.get_active_learn_prompts(TEST_SENDER)
        assert len(active) == 1  # Only lp2 is active; lp is completed
        assert active[0].id == lp2.id
