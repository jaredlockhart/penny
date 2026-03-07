"""Tests for tool call validation with missing required parameters."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools.fetch_news import FetchNewsTool
from penny.tools.news import NewsArticle
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
    """Tests for FetchNewsTool parameter handling (issue #660)."""

    def _make_news_tool(
        self, articles: list[NewsArticle] | None = None
    ) -> tuple[FetchNewsTool, AsyncMock]:
        """Build a FetchNewsTool with a mocked NewsTool. Returns (tool, mock_search)."""
        mock_news = MagicMock()
        mock_search = AsyncMock(return_value=articles or [])
        mock_news.search = mock_search
        return FetchNewsTool(news_tool=mock_news), mock_search

    def test_fetch_news_topic_not_in_required(self):
        """The 'topic' parameter must not be in required — prevents validation failures (#660)."""
        tool, _ = self._make_news_tool()
        required = tool.parameters.get("required", [])
        assert "topic" not in required

    def test_fetch_news_ollama_tool_topic_not_in_required(self):
        """Ollama tool format must not list 'topic' as required (#660)."""
        tool, _ = self._make_news_tool()
        ollama_tool = tool.to_ollama_tool()
        required = ollama_tool["function"]["parameters"].get("required", [])
        assert "topic" not in required

    @pytest.mark.asyncio
    async def test_fetch_news_executes_without_topic(self):
        """FetchNewsTool should succeed without 'topic', defaulting to 'top stories' (#660)."""
        article = NewsArticle(
            title="Top Story",
            description="A breaking story.",
            url="https://example.com/story",
            published_at=datetime(2026, 3, 7, 12, 0, 0),
            source_name="Example News",
        )
        tool, mock_search = self._make_news_tool(articles=[article])

        result = await tool.execute()  # no topic argument

        assert "Top Story" in result
        mock_search.assert_called_once_with(query_terms=["top stories"])

    @pytest.mark.asyncio
    async def test_fetch_news_executes_with_explicit_topic(self):
        """FetchNewsTool should use the provided topic when one is given."""
        article = NewsArticle(
            title="AI Article",
            description="Something about AI.",
            url="https://example.com/ai",
            published_at=datetime(2026, 3, 7, 12, 0, 0),
            source_name="Tech News",
        )
        tool, mock_search = self._make_news_tool(articles=[article])

        result = await tool.execute(topic="artificial intelligence")

        assert "AI Article" in result
        mock_search.assert_called_once_with(query_terms=["artificial intelligence"])
