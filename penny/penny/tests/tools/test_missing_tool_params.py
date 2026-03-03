"""Tests for tool call validation with missing required parameters."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools.base import ToolExecutor, ToolRegistry
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.models import ToolCall
from penny.tools.search import SearchTool

_IMAGE_MAX_RESULTS = int(RUNTIME_CONFIG_PARAMS["IMAGE_MAX_RESULTS"].default)
_IMAGE_TIMEOUT = RUNTIME_CONFIG_PARAMS["IMAGE_DOWNLOAD_TIMEOUT"].default


class TestFetchNewsToolSchema:
    """Tests for FetchNewsTool schema and description quality."""

    def test_description_guides_model_to_infer_topic(self):
        """FetchNewsTool description should tell model to infer topic from context."""
        mock_news_tool = MagicMock()
        tool = FetchNewsTool(mock_news_tool)
        desc = tool.description.lower()
        assert "infer" in desc or "context" in desc, (
            "Description should guide the model to infer topic from user's message context"
        )

    def test_description_says_ask_when_vague(self):
        """FetchNewsTool description should tell model to ask user when request is vague."""
        mock_news_tool = MagicMock()
        tool = FetchNewsTool(mock_news_tool)
        desc = tool.description.lower()
        assert "ask" in desc or "clarif" in desc, (
            "Description should tell the model to ask the user when the topic is unclear"
        )

    def test_topic_param_description_has_examples(self):
        """The topic parameter description should include concrete examples."""
        from typing import Any, cast

        mock_news_tool = MagicMock()
        tool = FetchNewsTool(mock_news_tool)
        props = cast(dict[str, Any], tool.parameters["properties"])
        topic_entry = cast(dict[str, Any], props["topic"])
        topic_desc = cast(str, topic_entry["description"]).lower()
        assert "e.g" in topic_desc or "example" in topic_desc or "spacex" in topic_desc, (
            "Topic parameter description should include examples to guide the model"
        )

    def test_topic_is_required(self):
        """Topic must be listed in the required parameters."""
        mock_news_tool = MagicMock()
        tool = FetchNewsTool(mock_news_tool)
        assert "topic" in tool.parameters.get("required", [])


class TestMissingTopicValidationError:
    """Tests for validation error message when fetch_news is called without topic."""

    @pytest.mark.asyncio
    async def test_missing_topic_error_guides_context_lookup(self):
        """Validation error for missing topic should guide model to determine value from context."""
        mock_news_tool = MagicMock()
        tool = FetchNewsTool(mock_news_tool)

        registry = ToolRegistry()
        registry.register(tool)
        executor = ToolExecutor(registry, timeout=5.0)

        result = await executor.execute(ToolCall(tool="fetch_news", arguments={}))

        assert result.error is not None
        error_lower = result.error.lower()
        assert "topic" in error_lower
        assert "context" in error_lower or "determine" in error_lower, (
            "Error message should guide the model to determine the value from context"
        )

    @pytest.mark.asyncio
    async def test_missing_topic_error_does_not_say_please_call_again(self):
        """Validation error should not use the old 'Please call the tool again' phrasing."""
        mock_news_tool = MagicMock()
        tool = FetchNewsTool(mock_news_tool)

        registry = ToolRegistry()
        registry.register(tool)
        executor = ToolExecutor(registry, timeout=5.0)

        result = await executor.execute(ToolCall(tool="fetch_news", arguments={}))

        assert result.error is not None
        assert "please call the tool again" not in result.error.lower(), (
            "Error message should not use the old phrase that encourages blind retrying"
        )

    @pytest.mark.asyncio
    async def test_fetch_news_with_topic_executes_successfully(self):
        """FetchNewsTool should execute successfully when topic is provided."""
        mock_news_tool = MagicMock()
        mock_news_tool.search = AsyncMock(return_value=[])
        tool = FetchNewsTool(mock_news_tool)

        registry = ToolRegistry()
        registry.register(tool)
        executor = ToolExecutor(registry, timeout=5.0)

        result = await executor.execute(ToolCall(tool="fetch_news", arguments={"topic": "SpaceX"}))

        assert result.error is None
        assert "SpaceX" in result.result or "No recent news" in result.result


class TestMissingToolParams:
    """Test handling of tool calls with missing required parameters."""

    @pytest.mark.asyncio
    async def test_search_tool_missing_query_raises_keyerror(self):
        """SearchTool.execute raises KeyError when 'query' parameter is missing."""
        tool = SearchTool(
            perplexity_api_key="test-key",
            image_max_results=_IMAGE_MAX_RESULTS,
            image_download_timeout=_IMAGE_TIMEOUT,
        )

        # Call execute with empty kwargs (missing required 'query' parameter)
        with pytest.raises(KeyError, match="query"):
            await tool.execute()

    @pytest.mark.asyncio
    async def test_agent_handles_missing_required_parameter(self, test_db, mock_ollama):
        """Agent should handle tool calls with missing required parameters gracefully."""
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
            max_steps=3,
        )

        # Track messages sent to the model to verify error handling
        messages_sent = []

        def handler(request: dict, count: int) -> dict:
            messages_sent.append(request["messages"])
            if count == 1:
                # First call: return malformed tool call with empty arguments
                return mock_ollama._make_tool_call_response(request, "search", {})
            # Second call: return final response after receiving error
            return mock_ollama._make_text_response(
                request, "I apologize, I need more information to search."
            )

        mock_ollama.set_response_handler(handler)

        # Agent should not crash - it should handle the error gracefully
        response = await agent.run("test prompt")

        # Verify that we got a response (not a crash)
        assert response.answer is not None
        assert "apologize" in response.answer.lower()

        # The error should have been sent back to the model as a tool result
        assert len(messages_sent) == 2  # Initial call + retry after error
        # The second call should include a TOOL role message with the error
        second_call_messages = messages_sent[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0
        # The error should mention the tool execution error
        error_content = tool_messages[0]["content"]
        assert "error" in error_content.lower()
        # Verify it mentions the missing parameter
        assert "query" in error_content.lower()
        assert "parameter" in error_content.lower()

        await agent.close()
