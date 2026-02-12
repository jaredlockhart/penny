"""Tests for tool call validation with missing required parameters."""

import pytest

from penny.agents.base import Agent
from penny.database import Database
from penny.tools.builtin import SearchTool


class TestMissingToolParams:
    """Test handling of tool calls with missing required parameters."""

    @pytest.mark.asyncio
    async def test_search_tool_missing_query_raises_keyerror(self):
        """SearchTool.execute raises KeyError when 'query' parameter is missing."""
        tool = SearchTool(perplexity_api_key="test-key")

        # Call execute with empty kwargs (missing required 'query' parameter)
        with pytest.raises(KeyError, match="query"):
            await tool.execute()

    @pytest.mark.asyncio
    async def test_agent_handles_missing_required_parameter(self, test_db, mock_ollama):
        """Agent should handle tool calls with missing required parameters gracefully."""
        db = Database(test_db)
        db.create_tables()

        search_tool = SearchTool(perplexity_api_key="test-key", db=db)

        agent = Agent(
            system_prompt="test",
            model="test-model",
            ollama_api_url="http://localhost:11434",
            tools=[search_tool],
            db=db,
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
