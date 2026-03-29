"""Tests for tool call validation with missing required parameters."""

from unittest.mock import AsyncMock

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.search import SearchTool


class TestMissingToolParams:
    """Test handling of tool calls with missing required parameters."""

    @pytest.mark.asyncio
    async def test_search_tool_missing_query_raises_validation_error(self):
        """SearchTool.execute raises ValidationError when 'query' parameter is missing."""
        from pydantic import ValidationError

        tool = SearchTool(perplexity_api_key="test-key")

        # Call execute with empty kwargs (missing required 'query' parameter)
        with pytest.raises(ValidationError, match="query"):
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
        search_tool = SearchTool(perplexity_api_key="test-key", db=db)

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
                request, "I need a search query to look that up for you."
            )

        mock_ollama.set_response_handler(handler)

        # Agent should not crash - it should handle the error gracefully
        response = await agent.run("test prompt", max_steps=3)

        # Verify that we got a response (not a crash)
        assert response.answer is not None
        assert "search query" in response.answer.lower()

        # The error should have been sent back to the model as a tool result
        assert len(messages_sent) == 2  # Initial call + retry after error
        # The second call should include a TOOL role message with the error
        second_call_messages = messages_sent[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0
        # The error should mention the tool execution error and missing key
        error_content = tool_messages[0]["content"]
        assert "error" in error_content.lower()
        assert "query" in error_content.lower()

        await agent.close()

    @pytest.mark.asyncio
    async def test_fetch_news_tool_uses_default_topic_when_omitted(self):
        """FetchNewsTool.execute uses 'top news' as default when topic is omitted."""
        mock_news = AsyncMock()
        mock_news.search.return_value = []
        tool = FetchNewsTool(news_tool=mock_news)

        # Call without topic — should not raise, should use default
        result = await tool.execute()

        assert result == "No recent news found for 'top news'."
        mock_news.search.assert_called_once_with(query_terms=["top news"])

    @pytest.mark.asyncio
    async def test_fetch_news_tool_not_in_required_params(self):
        """FetchNewsTool schema does not require topic — model can omit it safely."""
        mock_news = AsyncMock()
        mock_news.search.return_value = []
        tool = FetchNewsTool(news_tool=mock_news)

        required = tool.parameters.get("required", [])
        assert "topic" not in required

    @pytest.mark.asyncio
    async def test_agent_handles_fetch_news_without_topic(self, test_db, mock_ollama):
        """Agent succeeds when fetch_news is called without topic — uses default."""
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

        mock_news = AsyncMock()
        mock_news.search.return_value = []
        fetch_news_tool = FetchNewsTool(news_tool=mock_news)

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
            tools=[fetch_news_tool],
            db=db,
            config=config,
        )

        def handler(request: dict, count: int) -> dict:
            if count == 1:
                # Model calls fetch_news with no topic argument
                return mock_ollama._make_tool_call_response(request, "fetch_news", {})
            return mock_ollama._make_text_response(request, "No news found today.")

        mock_ollama.set_response_handler(handler)

        response = await agent.run("what's in the news?", max_steps=3)

        # Should succeed without a validation error round-trip
        assert response.answer == "No news found today."
        # NewsTool should have been called with the default topic
        mock_news.search.assert_called_once_with(query_terms=["top news"])

        await agent.close()
