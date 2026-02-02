"""Tool executor with safety guardrails."""

import asyncio
import logging

from penny.tools import ToolCall, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tools with validation and safety guardrails."""

    def __init__(self, registry: ToolRegistry, timeout: float = 30.0):
        """
        Initialize tool executor.

        Args:
            registry: Tool registry
            timeout: Maximum execution time per tool (seconds)
        """
        self.registry = registry
        self.timeout = timeout

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            Tool result
        """
        tool = self.registry.get(tool_call.tool)

        if tool is None:
            logger.error("Tool not found: %s", tool_call.tool)
            return ToolResult(
                tool=tool_call.tool,
                result=None,
                error=f"Tool '{tool_call.tool}' not found",
                id=tool_call.id,
            )

        try:
            logger.info("Executing tool: %s", tool_call.tool)
            logger.debug("Tool arguments: %s", tool_call.arguments)

            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute(**tool_call.arguments),
                timeout=self.timeout,
            )

            logger.info("Tool executed successfully: %s", tool_call.tool)
            logger.debug("Tool result: %s", result)

            return ToolResult(
                tool=tool_call.tool,
                result=result,
                error=None,
                id=tool_call.id,
            )

        except TimeoutError:
            logger.error("Tool execution timeout: %s", tool_call.tool)
            return ToolResult(
                tool=tool_call.tool,
                result=None,
                error=f"Tool execution timeout after {self.timeout}s",
                id=tool_call.id,
            )

        except Exception as e:
            logger.exception("Tool execution error: %s", tool_call.tool)
            return ToolResult(
                tool=tool_call.tool,
                result=None,
                error=f"Tool execution error: {str(e)}",
                id=tool_call.id,
            )
