"""Tests for handling tool calls with non-existent tool names and missing parameters."""

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.database import Database
from penny.llm import LlmClient
from penny.tools.base import Tool, ToolExecutor, ToolRegistry


class StubSearchTool(Tool):
    """Minimal stub tool for testing tool-not-found handling."""

    name = "search"
    description = "Search for information"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }

    async def execute(self, **kwargs):
        return "Mock search results for testing"


class TestToolNotFound:
    """Test handling of tool calls for tools that don't exist."""

    @pytest.mark.asyncio
    async def test_agent_returns_helpful_error_for_nonexistent_tool(self, test_db, mock_llm):
        """Agent returns helpful error listing available tools for non-existent tool."""
        db = Database(test_db)
        db.create_tables()

        config = Config(
            channel_type="signal",
            signal_number="+15551234567",
            signal_api_url="http://localhost:8080",
            discord_bot_token=None,
            discord_channel_id=None,
            llm_api_url="http://localhost:11434",
            llm_model="test-model",
            log_level="DEBUG",
            db_path=test_db,
        )
        search_tool = StubSearchTool()

        client = LlmClient(
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
                # First call: return tool call with non-existent tool name
                return mock_llm._make_tool_call_response(
                    request, "example_function_name", {"query": "test"}
                )
            # Second call: return final response after receiving error
            return mock_llm._make_text_response(request, "Let me use the correct search tool.")

        mock_llm.set_response_handler(handler)

        # Agent should not crash - it should handle the error gracefully
        response = await agent.run("test prompt", max_steps=3)

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


class StubDoneTool(Tool):
    """Stub tool with two required typed+described parameters."""

    name = "stub_done"
    description = "Signal completion"
    parameters = {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "True if the cycle succeeded.",
            },
            "summary": {
                "type": "string",
                "description": "One-sentence description of what was done.",
            },
        },
        "required": ["success", "summary"],
    }

    async def execute(self, **kwargs):
        return "done"


class TestMissingRequiredParameters:
    """Validation error messages include parameter type and description hints."""

    def test_missing_params_error_includes_type_and_description(self):
        """Error message includes type and description for each missing parameter."""
        registry = ToolRegistry()
        tool = StubDoneTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        error = executor._validate_arguments(tool, {})

        assert error is not None
        assert "success" in error
        assert "boolean" in error
        assert "True if the cycle succeeded" in error
        assert "summary" in error
        assert "string" in error
        assert "One-sentence description" in error

    def test_missing_params_error_only_lists_absent_params(self):
        """Only the actually-missing parameter appears in the error."""
        registry = ToolRegistry()
        tool = StubDoneTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        error = executor._validate_arguments(tool, {"success": True})

        assert error is not None
        assert "summary" in error
        assert "success" not in error

    def test_no_error_when_all_required_params_present(self):
        """Returns None when all required parameters are provided."""
        registry = ToolRegistry()
        tool = StubDoneTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        error = executor._validate_arguments(tool, {"success": True, "summary": "done"})

        assert error is None

    @pytest.mark.asyncio
    async def test_agent_sends_hint_rich_error_to_model_on_missing_params(self, test_db, mock_llm):
        """Validation error with type hints is fed back to the model for retry."""
        db = Database(test_db)
        db.create_tables()

        config = Config(
            channel_type="signal",
            signal_number="+15551234567",
            signal_api_url="http://localhost:8080",
            discord_bot_token=None,
            discord_channel_id=None,
            llm_api_url="http://localhost:11434",
            llm_model="test-model",
            log_level="DEBUG",
            db_path=test_db,
        )
        tool = StubDoneTool()
        client = LlmClient(
            api_url="http://localhost:11434",
            model="test-model",
            db=db,
            max_retries=1,
            retry_delay=0.1,
        )
        agent = Agent(
            system_prompt="test",
            model_client=client,
            tools=[tool],
            db=db,
            config=config,
        )

        messages_sent = []

        def handler(request: dict, count: int) -> dict:
            messages_sent.append(request["messages"])
            if count == 1:
                # Call done with no arguments
                return mock_llm._make_tool_call_response(request, "stub_done", {})
            return mock_llm._make_text_response(request, "Fixed.")

        mock_llm.set_response_handler(handler)

        await agent.run("test", max_steps=3)

        # The error fed back to the model must include type hints
        second_call_messages = messages_sent[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0
        error_content = tool_messages[0]["content"]
        assert "boolean" in error_content
        assert "string" in error_content

        await agent.close()


class StubCollectionWriteTool(Tool):
    """Stub for collection_write to test malformed-name sanitization."""

    name = "collection_write"
    description = "Write an entry to a collection."
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Entry key"},
            "content": {"type": "string", "description": "Entry content"},
        },
        "required": ["key", "content"],
    }

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return "written"


class TestToolNameSanitization:
    """LlmClient strips prose fragments from malformed tool names before dispatch."""

    def test_unicode_ellipsis_truncates_tool_name(self):
        """collection_write…Wait is sanitized to collection_write."""
        from penny.llm import LlmClient

        result = LlmClient._sanitize_tool_name("collection_write…Wait")
        assert result == "collection_write"

    def test_ascii_ellipsis_mid_name_truncates(self):
        """collection_write...Wait is sanitized to collection_write."""
        from penny.llm import LlmClient

        result = LlmClient._sanitize_tool_name("collection_write...Wait")
        assert result == "collection_write"

    def test_leading_ascii_ellipsis_yields_empty_string(self):
        """...Wait (no valid prefix) sanitizes to empty string."""
        from penny.llm import LlmClient

        result = LlmClient._sanitize_tool_name("...Wait")
        assert result == ""

    def test_clean_tool_name_passes_through_unchanged(self):
        """Valid identifier names are not modified."""
        from penny.llm import LlmClient

        assert LlmClient._sanitize_tool_name("collection_write") == "collection_write"
        assert LlmClient._sanitize_tool_name("read_similar") == "read_similar"
        assert LlmClient._sanitize_tool_name("browse") == "browse"

    def test_leading_trailing_non_identifier_chars_stripped(self):
        """Non-identifier edge characters are stripped."""
        from penny.llm import LlmClient

        assert LlmClient._sanitize_tool_name(" search ") == "search"
        assert LlmClient._sanitize_tool_name("?search?") == "search"

    @pytest.mark.asyncio
    async def test_malformed_unicode_ellipsis_name_routes_to_real_tool(self, test_db, mock_llm):
        """When LLM emits 'collection_write…Wait', sanitization routes to collection_write."""
        db = Database(test_db)
        db.create_tables()

        config = Config(
            channel_type="signal",
            signal_number="+15551234567",
            signal_api_url="http://localhost:8080",
            discord_bot_token=None,
            discord_channel_id=None,
            llm_api_url="http://localhost:11434",
            llm_model="test-model",
            log_level="DEBUG",
            db_path=test_db,
        )
        write_tool = StubCollectionWriteTool()
        client = LlmClient(
            api_url="http://localhost:11434",
            model="test-model",
            db=db,
            max_retries=1,
            retry_delay=0.1,
        )
        agent = Agent(
            system_prompt="test",
            model_client=client,
            tools=[write_tool],
            db=db,
            config=config,
        )

        def handler(request: dict, count: int) -> dict:
            if count == 1:
                # LLM emits malformed name with unicode ellipsis suffix
                return mock_llm._make_tool_call_response(
                    request, "collection_write…Wait", {"key": "k", "content": "v"}
                )
            return mock_llm._make_text_response(request, "Done.")

        mock_llm.set_response_handler(handler)

        response = await agent.run("test", max_steps=3)

        assert response.answer is not None
        # The sanitized tool name must have executed collection_write successfully
        assert len(write_tool.calls) == 1
        assert write_tool.calls[0] == {"key": "k", "content": "v"}

        await agent.close()
