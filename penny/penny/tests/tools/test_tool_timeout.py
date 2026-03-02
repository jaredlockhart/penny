"""Tests for tool execution timeout configuration."""

import asyncio

import pytest

from penny.agents.base import Agent
from penny.config import Config
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools import ToolCall, ToolExecutor, ToolRegistry
from penny.tools.base import Tool


class SlowTool(Tool):
    """Test tool that sleeps for a configurable duration."""

    name = "slow_tool"
    description = "A tool that takes a long time"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, sleep_duration: float):
        self.sleep_duration = sleep_duration

    async def execute(self, **kwargs):
        """Sleep for the configured duration."""
        await asyncio.sleep(self.sleep_duration)
        return "completed"


class FlakeyTool(Tool):
    """Test tool that times out on the first N calls, then succeeds."""

    name = "flakey_tool"
    description = "A tool that fails the first few times"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, fail_count: int, slow_duration: float = 1.0):
        self.fail_count = fail_count
        self.slow_duration = slow_duration
        self.call_count = 0

    async def execute(self, **kwargs):
        """Sleep long enough to timeout on the first fail_count calls, then succeed."""
        self.call_count += 1
        if self.call_count <= self.fail_count:
            await asyncio.sleep(self.slow_duration)
        return "completed"


class TestToolTimeout:
    """Test tool execution timeout behavior."""

    @pytest.mark.asyncio
    async def test_tool_timeout_enforced(self):
        """Tool execution should timeout after configured duration."""
        registry = ToolRegistry()
        slow_tool = SlowTool(sleep_duration=1.0)
        registry.register(slow_tool)

        # Set timeout to 0.1 seconds (tool takes 1s, well past the timeout)
        executor = ToolExecutor(registry, timeout=0.1)

        tool_call = ToolCall(tool="slow_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.error is not None
        assert "timeout" in result.error.lower()
        assert result.result is None

    @pytest.mark.asyncio
    async def test_tool_completes_within_timeout(self):
        """Tool execution should succeed if it completes within timeout."""
        registry = ToolRegistry()
        fast_tool = SlowTool(sleep_duration=0.1)
        registry.register(fast_tool)

        # Set timeout to 2 seconds
        executor = ToolExecutor(registry, timeout=2.0)

        tool_call = ToolCall(tool="slow_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.error is None
        assert result.result == "completed"

    @pytest.mark.asyncio
    async def test_tool_retried_on_timeout(self):
        """Tool should be retried once on timeout and succeed on the second attempt."""
        registry = ToolRegistry()
        # Tool times out on first call (sleeps 1s > 0.1s timeout), succeeds on second
        flakey_tool = FlakeyTool(fail_count=1, slow_duration=1.0)
        registry.register(flakey_tool)

        executor = ToolExecutor(registry, timeout=0.1, max_retries=1)
        tool_call = ToolCall(tool="flakey_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.error is None
        assert result.result == "completed"
        assert flakey_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_exhausts_retries_and_returns_error(self):
        """Tool should return error after exhausting all retries."""
        registry = ToolRegistry()
        # Tool always times out (sleeps 1s > 0.1s timeout)
        slow_tool = SlowTool(sleep_duration=1.0)
        registry.register(slow_tool)

        executor = ToolExecutor(registry, timeout=0.1, max_retries=1)
        tool_call = ToolCall(tool="slow_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.error is not None
        assert "timeout" in result.error.lower()
        assert result.result is None

    @pytest.mark.asyncio
    async def test_tool_no_retry_on_non_timeout_error(self):
        """Non-timeout errors should not trigger a retry."""

        class ErrorTool(Tool):
            name = "error_tool"
            description = "A tool that raises an error"
            parameters = {"type": "object", "properties": {}}
            call_count = 0

            async def execute(self, **kwargs):
                self.call_count += 1
                raise ValueError("something went wrong")

        registry = ToolRegistry()
        error_tool = ErrorTool()
        registry.register(error_tool)

        executor = ToolExecutor(registry, timeout=5.0, max_retries=1)
        tool_call = ToolCall(tool="error_tool", arguments={})
        result = await executor.execute(tool_call)

        assert result.error is not None
        assert "something went wrong" in result.error
        # Should NOT retry on non-timeout errors
        assert error_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_agent_uses_configured_timeout(self, test_db):
        """Agent should use tool_timeout parameter when creating ToolExecutor."""
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
        # Create agent with custom timeout
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
            tools=[],
            db=db,
            config=config,
            tool_timeout=90.0,
        )

        # Check that the ToolExecutor was initialized with the correct timeout
        assert agent._tool_executor.timeout == 90.0

        await agent.close()
