"""Tests for handling tool calls with non-existent tool names."""

import pytest

from penny.agents.base import Agent
from penny.database import Database
from penny.tools.builtin import SearchTool


class TestToolNotFound:
    """Test handling of tool calls for tools that don't exist."""

    @pytest.mark.asyncio
    async def test_agent_returns_helpful_error_for_nonexistent_tool(self, test_db, mock_ollama):
        """Agent returns helpful error listing available tools for non-existent tool."""
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
                # First call: return tool call with non-existent tool name
                return mock_ollama._make_tool_call_response(
                    request, "example_function_name", {"query": "test"}
                )
            # Second call: return final response after receiving error
            return mock_ollama._make_text_response(request, "Let me use the correct search tool.")

        mock_ollama.set_response_handler(handler)

        # Agent should not crash - it should handle the error gracefully
        response = await agent.run("test prompt")

        # Verify that we got a response (not a crash)
        assert response.answer is not None

        # The error should have been sent back to the model as a tool result
        assert len(messages_sent) == 2  # Initial call + retry after error
        # The second call should include a TOOL role message with the error
        second_call_messages = messages_sent[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0

        # The error should list available tools
        error_content = tool_messages[0]["content"]
        assert "not found" in error_content.lower()
        assert "available" in error_content.lower()
        assert "search" in error_content.lower()  # The actual tool name

        await agent.close()
