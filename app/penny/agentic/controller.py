"""Agentic controller loop using Ollama SDK."""

import logging
from collections.abc import Callable

from penny.agentic.models import ChatMessage, ControllerResponse, MessageRole
from penny.agentic.tool_executor import ToolExecutor
from penny.ollama import OllamaClient
from penny.tools import ToolCall, ToolRegistry
from penny.tools.models import SearchResult

logger = logging.getLogger(__name__)


class AgenticController:
    """Controls the agentic loop with native Ollama tool calling."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        tool_registry: ToolRegistry,
        max_steps: int = 5,
    ):
        """
        Initialize controller.

        Args:
            ollama_client: Ollama client for LLM calls
            tool_registry: Registry of available tools
            max_steps: Maximum agentic steps before forcing answer
        """
        self.ollama = ollama_client
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)
        self.max_steps = max_steps

    def _build_messages(
        self,
        current_message: str,
        system_prompt: str | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        """
        Build message list for Ollama chat API.

        Args:
            current_message: Current user message
            system_prompt: Optional system prompt
            history: Optional list of (role, content) tuples for conversation history

        Returns:
            List of message dicts for Ollama chat API
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt).to_dict())

        if history:
            for role, content in history:
                messages.append(ChatMessage(role=MessageRole(role), content=content).to_dict())

        messages.append(ChatMessage(role=MessageRole.USER, content=current_message).to_dict())

        return messages

    async def run(
        self,
        current_message: str,
        system_prompt: str | None = None,
        history: list[tuple[str, str]] | None = None,
        on_step: Callable | None = None,
    ) -> ControllerResponse:
        """
        Run the agentic loop with tool calling.

        Args:
            current_message: Current user message
            system_prompt: Optional system prompt for special instructions
            history: Optional list of (role, content) tuples for conversation history
            on_step: Optional callback called at the start of each iteration

        Returns:
            ControllerResponse with answer and optional thinking
        """
        # Build messages
        messages = self._build_messages(current_message, system_prompt, history)

        # Get tools in Ollama format
        tools = self.tool_registry.get_ollama_tools()
        logger.debug("Using %d tools", len(tools))

        # Collect image attachments from tool results
        attachments: list[str] = []
        # Track tools that have been called to prevent repeat loops
        called_tools: set[str] = set()

        # Agentic loop
        for step in range(self.max_steps):
            logger.info("Agentic step %d/%d", step + 1, self.max_steps)

            # Call step callback if provided (e.g., to refresh typing indicator)
            if on_step:
                await on_step(step)

            # Get model response
            try:
                response = await self.ollama.chat(messages=messages, tools=tools)
            except Exception as e:
                logger.error("Error calling Ollama: %s", e)
                return ControllerResponse(
                    answer="Sorry, I encountered an error communicating with the model."
                )

            # Check for tool calls
            if response.has_tool_calls:
                logger.info(
                    "Model requested %d tool call(s)", len(response.message.tool_calls or [])
                )

                # Add assistant message to history
                messages.append(response.message.model_dump())

                # Execute each tool call
                for ollama_tool_call in response.message.tool_calls or []:
                    function = ollama_tool_call.get("function", {})
                    tool_name = function.get("name", "")
                    arguments = function.get("arguments", {})

                    # Skip tools that have already been called
                    if tool_name in called_tools:
                        logger.info("Skipping repeat call to tool: %s", tool_name)
                        result_str = (
                            "tool already called, do not call it again. write your response now."
                        )
                        messages.append(
                            ChatMessage(role=MessageRole.TOOL, content=result_str).to_dict()
                        )
                        continue

                    logger.info("Executing tool: %s", tool_name)
                    called_tools.add(tool_name)

                    tool_call = ToolCall(tool=tool_name, arguments=arguments)
                    tool_result = await self.tool_executor.execute(tool_call)

                    if tool_result.error:
                        result_str = f"Error: {tool_result.error}"
                    elif isinstance(tool_result.result, SearchResult):
                        result_str = tool_result.result.text
                        if tool_result.result.image_base64:
                            attachments.append(tool_result.result.image_base64)
                    else:
                        result_str = str(tool_result.result)
                    logger.debug("Tool result: %s", result_str[:200])

                    messages.append(
                        ChatMessage(role=MessageRole.TOOL, content=result_str).to_dict()
                    )

                continue

            # No tool calls - this is the final answer
            content = response.content.strip()

            if not content:
                logger.error("Model returned empty content!")
                return ControllerResponse(answer="Sorry, the model generated an empty response.")

            thinking = response.thinking or response.message.thinking

            if thinking:
                logger.info("Extracted thinking text (length: %d)", len(thinking))

            logger.info("Got final answer (length: %d)", len(content))
            return ControllerResponse(answer=content, thinking=thinking, attachments=attachments)

        # Max steps reached
        logger.warning("Max steps reached without final answer")
        return ControllerResponse(
            answer="Sorry, I couldn't complete that request within the allowed steps."
        )
