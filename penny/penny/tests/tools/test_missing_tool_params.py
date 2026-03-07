"""Tests for tool call validation with missing required parameters."""

from unittest.mock import AsyncMock

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.search import SearchTool

_IMAGE_MAX_RESULTS = int(RUNTIME_CONFIG_PARAMS["IMAGE_MAX_RESULTS"].default)
_IMAGE_TIMEOUT = RUNTIME_CONFIG_PARAMS["IMAGE_DOWNLOAD_TIMEOUT"].default


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
            ollama_model="test-model",
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
            model_client=client,
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


class TestFetchNewsTool:
    """Test FetchNewsTool parameter handling."""

    def test_topic_not_required_in_schema(self):
        """FetchNewsTool schema must NOT list 'topic' in required."""
        mock_news_tool = AsyncMock()
        tool = FetchNewsTool(mock_news_tool)
        assert "required" not in tool.parameters or "topic" not in tool.parameters.get(
            "required", []
        )

    @pytest.mark.asyncio
    async def test_execute_defaults_to_top_stories_when_topic_omitted(self):
        """When topic is omitted, execute() searches for 'top stories'."""
        mock_news_tool = AsyncMock()
        mock_news_tool.search.return_value = []
        tool = FetchNewsTool(mock_news_tool)

        result = await tool.execute()

        mock_news_tool.search.assert_called_once_with(query_terms=["top stories"])
        assert "top stories" in result

    @pytest.mark.asyncio
    async def test_execute_uses_provided_topic(self):
        """When topic is provided, execute() uses it."""
        mock_news_tool = AsyncMock()
        mock_news_tool.search.return_value = []
        tool = FetchNewsTool(mock_news_tool)

        result = await tool.execute(topic="artificial intelligence")

        mock_news_tool.search.assert_called_once_with(query_terms=["artificial intelligence"])
        assert "artificial intelligence" in result
