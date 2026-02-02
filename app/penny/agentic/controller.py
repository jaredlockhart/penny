"""Agentic controller loop using Ollama SDK."""

import json
import logging

from penny.agentic.models import ChatMessage, ControllerResponse, MessageRole
from penny.agentic.tool_executor import ToolExecutor
from penny.ollama import OllamaClient
from penny.tools import ToolCall, ToolRegistry

logger = logging.getLogger(__name__)


class AgenticController:
    """Controls the agentic loop with native Ollama tool calling."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        tool_registry: ToolRegistry,
        db,
        max_steps: int = 5,
    ):
        """
        Initialize controller.

        Args:
            ollama_client: Ollama client for LLM calls
            tool_registry: Registry of available tools
            db: Database instance for retrieving memories
            max_steps: Maximum agentic steps before forcing answer
        """
        self.ollama = ollama_client
        self.tool_registry = tool_registry
        self.db = db
        self.tool_executor = ToolExecutor(tool_registry)
        self.max_steps = max_steps

    def _build_messages(
        self, history: list, current_message: str, system_prompt: str | None = None
    ) -> list[dict]:
        """
        Build message list from conversation history.

        Args:
            history: Conversation history (Message objects)
            current_message: Current user message
            system_prompt: Optional system prompt to prepend (for classification, etc.)

        Returns:
            List of message dicts for Ollama chat API
        """
        messages = []

        # Add optional system prompt (for special tasks like classification)
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt).to_dict())

        # Add system message with long-term memories (if any exist)
        memories = self.db.get_all_memories()
        if memories:
            memory_text = "Long-term memories:\n" + "\n".join(f"- {m.content}" for m in memories)
            messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=memory_text).to_dict()
            )

        # Add history
        for msg in history:
            role = MessageRole.USER if msg.direction == "incoming" else MessageRole.ASSISTANT
            # Only add first chunk to avoid duplication
            if msg.chunk_index is None or msg.chunk_index == 0:
                messages.append(ChatMessage(role=role, content=msg.content).to_dict())

        # Add current message
        messages.append(
            ChatMessage(role=MessageRole.USER, content=current_message).to_dict()
        )

        logger.debug("Built %d messages from history", len(messages))
        return messages

    async def run(
        self, history: list, current_message: str, system_prompt: str | None = None
    ) -> ControllerResponse:
        """
        Run the agentic loop with tool calling.

        Args:
            history: Conversation history (Message objects)
            current_message: Current user message
            system_prompt: Optional system prompt for special instructions

        Returns:
            ControllerResponse with answer and optional thinking
        """
        # Build messages
        messages = self._build_messages(history, current_message, system_prompt)

        # Get tools in Ollama format
        tools = self.tool_registry.get_ollama_tools()
        logger.debug("Using %d tools", len(tools))

        # Agentic loop
        for step in range(self.max_steps):
            logger.info("Agentic step %d/%d", step + 1, self.max_steps)

            # Get model response
            try:
                response = await self.ollama.chat(messages=messages, tools=tools)
            except Exception as e:
                logger.error("Error calling Ollama: %s", e)
                return ControllerResponse(
                    answer="Sorry, I encountered an error communicating with the model."
                )

            message = response.get("message", {})

            # Check for tool calls (handle None values)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                logger.info("Model requested %d tool call(s)", len(tool_calls))

                # Add assistant message to history
                messages.append(message)

                # Execute each tool call
                for ollama_tool_call in tool_calls:
                    function = ollama_tool_call.get("function", {})
                    tool_name = function.get("name", "")
                    arguments = function.get("arguments", {})

                    logger.info("Executing tool: %s", tool_name)

                    # Convert to our ToolCall format
                    tool_call = ToolCall(tool=tool_name, arguments=arguments)

                    # Execute tool
                    tool_result = await self.tool_executor.execute(tool_call)

                    # Format result
                    if tool_result.error:
                        result_str = f"Error: {tool_result.error}"
                    else:
                        result_str = str(tool_result.result)

                    logger.debug("Tool result: %s", result_str[:200])

                    # Add tool result to messages
                    messages.append(
                        ChatMessage(role=MessageRole.TOOL, content=result_str).to_dict()
                    )

                # Continue loop to get final answer
                continue

            # No tool calls - this is the final answer
            content = message.get("content", "").strip()

            if not content:
                logger.error("Model returned empty content!")
                return ControllerResponse(
                    answer="Sorry, the model generated an empty response."
                )

            # Extract thinking if present - check multiple possible locations
            thinking = (
                response.get("thinking")  # Top level in response
                or message.get("thinking")  # Inside message object
                or message.get("reasoning")  # Alternative field name
            )

            if thinking:
                logger.info("Extracted thinking text (length: %d)", len(thinking))
            else:
                logger.debug("No thinking text found. Response keys: %s, Message keys: %s",
                           list(response.keys()), list(message.keys()))

            logger.info("Got final answer (length: %d)", len(content))
            return ControllerResponse(answer=content, thinking=thinking)

        # Max steps reached
        logger.warning("Max steps reached without final answer")
        return ControllerResponse(
            answer="Sorry, I couldn't complete that request within the allowed steps."
        )
