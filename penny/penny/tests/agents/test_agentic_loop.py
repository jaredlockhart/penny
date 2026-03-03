"""Tests for agentic loop changes: reasoning, last step, and _after_step hook."""

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools.search import SearchTool

_IMAGE_MAX_RESULTS = int(RUNTIME_CONFIG_PARAMS["IMAGE_MAX_RESULTS"].default)
_IMAGE_TIMEOUT = RUNTIME_CONFIG_PARAMS["IMAGE_DOWNLOAD_TIMEOUT"].default


def _make_agent(test_db, mock_ollama, *, max_steps=3):
    """Create a minimal Agent for loop testing."""
    db = Database(test_db)
    db.create_tables()
    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://localhost:8080",
        discord_bot_token=None,
        discord_channel_id=None,
        ollama_api_url="http://localhost:11434",
        ollama_foreground_model="test-model",
        ollama_background_model="test-model",
        perplexity_api_key=None,
        log_level="DEBUG",
        db_path=test_db,
    )
    search_tool = SearchTool(
        perplexity_api_key="test-key",
        db=db,
        image_max_results=_IMAGE_MAX_RESULTS,
        image_download_timeout=_IMAGE_TIMEOUT,
    )
    client = OllamaClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.1,
    )
    agent = Agent(
        system_prompt="test",
        background_model_client=client,
        foreground_model_client=client,
        tools=[search_tool],
        db=db,
        config=config,
        max_steps=max_steps,
    )
    return agent, db


class TestReasoningStripped:
    """Test that reasoning is popped from tool arguments and stored on the record."""

    @pytest.mark.asyncio
    async def test_reasoning_captured_on_tool_call_record(self, test_db, mock_ollama):
        """Reasoning from tool call args is stored on ToolCallRecord."""
        agent, db = _make_agent(test_db, mock_ollama)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(
                    request,
                    "search",
                    {"query": "weather", "reasoning": "User asked about weather"},
                )
            return mock_ollama._make_text_response(request, "here's the weather!")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("what's the weather?")
        assert response.answer == "here's the weather!"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].reasoning == "User asked about weather"
        # reasoning should NOT be in the arguments dict
        assert "reasoning" not in response.tool_calls[0].arguments

        await agent.close()

    @pytest.mark.asyncio
    async def test_reasoning_none_when_not_provided(self, test_db, mock_ollama):
        """ToolCallRecord.reasoning is None when model doesn't provide it."""
        agent, db = _make_agent(test_db, mock_ollama)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(request, "search", {"query": "weather"})
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].reasoning is None

        await agent.close()


class TestLastStepToolRemoval:
    """Test that on the final step, tools are removed so the model must produce text."""

    @pytest.mark.asyncio
    async def test_final_step_has_no_tools(self, test_db, mock_ollama):
        """On the last step, the model is called without tools."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=2)

        def handler(request, count):
            if count == 1:
                # Step 1: model makes a tool call
                return mock_ollama._make_tool_call_response(request, "search", {"query": "test"})
            # Step 2 (final): model must produce text — verify no tools sent
            return mock_ollama._make_text_response(request, "final answer")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert response.answer == "final answer"

        # Step 1 should have tools, step 2 should not
        assert mock_ollama.requests[0]["tools"] is not None
        assert len(mock_ollama.requests[0]["tools"]) > 0
        assert mock_ollama.requests[1]["tools"] is None

        await agent.close()


class TestRepeatCallGuard:
    """Test that repeat tool calls are blocked by args, not just name."""

    @pytest.mark.asyncio
    async def test_same_tool_different_args_allowed(self, test_db, mock_ollama):
        """Calling the same tool with different arguments is allowed."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=4)

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "first topic"}
                )
            if count == 2:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "second topic"}
                )
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert response.answer == "done"
        # Both searches should have executed
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].arguments["query"] == "first topic"
        assert response.tool_calls[1].arguments["query"] == "second topic"

        await agent.close()

    @pytest.mark.asyncio
    async def test_same_tool_same_args_blocked(self, test_db, mock_ollama):
        """Calling the same tool with identical arguments is blocked."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        def handler(request, count):
            if count <= 2:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "same query"}
                )
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("test")
        assert response.answer == "done"
        # Only first call should have executed
        assert len(response.tool_calls) == 1

        await agent.close()


class TestEmptyContentRetry:
    """Test that empty model responses are retried before falling back to the error string."""

    @pytest.mark.asyncio
    async def test_empty_response_retried_and_recovers(self, test_db, mock_ollama):
        """Agent retries when model returns empty content, succeeds on second attempt.

        Uses the agentic loop path (use_tools=True) so _call_model_with_xml_retry is invoked.
        """
        agent, db = _make_agent(test_db, mock_ollama)

        def handler(request, count):
            if count == 1:
                # First attempt: empty content (thinking-only response, no tool calls)
                return mock_ollama._make_text_response(request, "")
            return mock_ollama._make_text_response(request, "recovered answer")

        mock_ollama.set_response_handler(handler)

        # use_tools=True (default) so the agentic loop and _call_model_with_xml_retry are used
        response = await agent.run("test")
        assert response.answer == "recovered answer"
        # Two calls: one empty (retry triggered), one successful
        assert len(mock_ollama.requests) == 2

        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_response_all_retries_exhausted_returns_fallback(
        self, test_db, mock_ollama
    ):
        """Agent returns fallback error string when all retries yield empty content."""
        from penny.responses import PennyResponse

        agent, db = _make_agent(test_db, mock_ollama)

        def handler(request, count):
            return mock_ollama._make_text_response(request, "")

        mock_ollama.set_response_handler(handler)

        # use_tools=True (default) so the agentic loop and _call_model_with_xml_retry are used
        response = await agent.run("test")
        assert response.answer == PennyResponse.AGENT_EMPTY_RESPONSE

        await agent.close()


class TestAfterStepHook:
    """Test the _after_step hook fires after tool calls."""

    @pytest.mark.asyncio
    async def test_after_step_called_with_step_records(self, test_db, mock_ollama):
        """_after_step receives only the records from the current step."""
        agent, db = _make_agent(test_db, mock_ollama, max_steps=3)

        captured_step_records = []

        async def capture_after_step(step_records, messages):
            captured_step_records.append(list(step_records))

        agent._after_step = capture_after_step

        def handler(request, count):
            if count == 1:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "first", "reasoning": "step 1 reason"}
                )
            if count == 2:
                return mock_ollama._make_tool_call_response(
                    request, "search", {"query": "second", "reasoning": "step 2 reason"}
                )
            return mock_ollama._make_text_response(request, "done")

        mock_ollama.set_response_handler(handler)
        agent.allow_repeat_tools = True

        response = await agent.run("test")
        assert response.answer == "done"

        # Two steps with tool calls → two _after_step calls
        assert len(captured_step_records) == 2
        assert len(captured_step_records[0]) == 1
        assert captured_step_records[0][0].reasoning == "step 1 reason"
        assert len(captured_step_records[1]) == 1
        assert captured_step_records[1][0].reasoning == "step 2 reason"

        await agent.close()
