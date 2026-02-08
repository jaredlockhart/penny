"""Base classes for tools."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from penny.tools.models import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for tools."""

    name: str
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool result (will be serialized to string for model)
        """
        pass

    def to_definition(self) -> ToolDefinition:
        """Convert to tool definition for prompt."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert to Ollama tool calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions for prompt building."""
        return [tool.to_definition() for tool in self._tools.values()]

    def get_ollama_tools(self) -> list[dict[str, Any]]:
        """Get all tools in Ollama format for tool calling."""
        return [tool.to_ollama_tool() for tool in self._tools.values()]


class ToolExecutor:
    """Executes tools with timeout and error handling."""

    def __init__(self, registry: ToolRegistry, timeout: float = 30.0):
        self.registry = registry
        self.timeout = timeout

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
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
