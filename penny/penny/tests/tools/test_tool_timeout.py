"""Tests for tool execution timeout configuration."""

import asyncio

import pytest

from penny.agent.base import Agent
from penny.database import Database
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
    async def test_agent_uses_configured_timeout(self, test_db):
        """Agent should use tool_timeout parameter when creating ToolExecutor."""
        db = Database(test_db)
        db.create_tables()

        # Create agent with custom timeout
        agent = Agent(
            system_prompt="test",
            model="test-model",
            ollama_api_url="http://localhost:11434",
            tools=[],
            db=db,
            tool_timeout=90.0,
        )

        # Check that the ToolExecutor was initialized with the correct timeout
        assert agent._tool_executor.timeout == 90.0

        await agent.close()
