"""Base Agent class with agentic loop."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from penny.agent.models import ChatMessage, ControllerResponse, MessageRole
from penny.database import Database
from penny.ollama import OllamaClient
from penny.tools import Tool, ToolCall, ToolExecutor, ToolRegistry
from penny.tools.models import SearchResult

logger = logging.getLogger(__name__)


class Agent:
    """
    AI agent with a specific persona and capabilities.

    Each Agent instance owns its own OllamaClient (for model isolation)
    and can have optional tools for agentic behavior.
    """

    _instances: list[Agent] = []

    @property
    def name(self) -> str:
        """Task name for logging. Override in subclasses."""
        return self.__class__.__name__

    async def execute(self) -> bool:
        """
        Execute a scheduled task. Override in subclasses.

        Returns:
            True if work was done, False otherwise
        """
        return False

    def __init__(
        self,
        system_prompt: str,
        model: str,
        ollama_api_url: str,
        tools: list[Tool],
        db: Database,
        max_steps: int = 5,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        tool_timeout: float = 60.0,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools
        self.db = db
        self.max_steps = max_steps

        self._ollama_client = OllamaClient(
            api_url=ollama_api_url,
            model=model,
            db=db,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        self._tool_registry = ToolRegistry()
        for tool in self.tools:
            self._tool_registry.register(tool)

        self._tool_executor = ToolExecutor(self._tool_registry, timeout=tool_timeout)

        Agent._instances.append(self)

        logger.info(
            "Initialized agent: model=%s, tools=%d, max_steps=%d",
            model,
            len(self.tools),
            max_steps,
        )

    def _build_messages(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        """Build message list for Ollama chat API."""
        messages = []

        now = datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M %p UTC")
        system_content = f"Current date and time: {now}\n\n{self.system_prompt}"
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content).to_dict())

        if history:
            for role, content in history:
                messages.append(ChatMessage(role=MessageRole(role), content=content).to_dict())

        messages.append(ChatMessage(role=MessageRole.USER, content=prompt).to_dict())

        return messages

    async def run(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
    ) -> ControllerResponse:
        """
        Run the agent with a prompt.

        Args:
            prompt: The user message/prompt to respond to
            history: Optional conversation history as (role, content) tuples

        Returns:
            ControllerResponse with answer, thinking, and attachments
        """
        messages = self._build_messages(prompt, history)
        tools = self._tool_registry.get_ollama_tools()
        logger.debug("Using %d tools", len(tools))

        attachments: list[str] = []
        source_urls: list[str] = []
        called_tools: set[str] = set()

        for step in range(self.max_steps):
            logger.info("Agent step %d/%d", step + 1, self.max_steps)

            try:
                response = await self._ollama_client.chat(messages=messages, tools=tools)
            except Exception as e:
                logger.error("Error calling Ollama: %s", e)
                return ControllerResponse(
                    answer="Sorry, I encountered an error communicating with the model."
                )

            if response.has_tool_calls:
                logger.info(
                    "Model requested %d tool call(s)", len(response.message.tool_calls or [])
                )

                messages.append(response.message.to_input_message())

                for ollama_tool_call in response.message.tool_calls or []:
                    tool_name = ollama_tool_call.function.name
                    arguments = ollama_tool_call.function.arguments

                    if tool_name in called_tools:
                        logger.info("Skipping repeat call to tool: %s", tool_name)
                        result_str = (
                            "Tool already called. DO NOT search again. Write your response NOW."
                        )
                        messages.append(
                            ChatMessage(role=MessageRole.TOOL, content=result_str).to_dict()
                        )
                        continue

                    logger.info("Executing tool: %s", tool_name)
                    called_tools.add(tool_name)

                    tool_call = ToolCall(tool=tool_name, arguments=arguments)
                    tool_result = await self._tool_executor.execute(tool_call)

                    if tool_result.error:
                        result_str = f"Error: {tool_result.error}"
                    elif isinstance(tool_result.result, SearchResult):
                        result_str = tool_result.result.text
                        if tool_result.result.urls:
                            source_urls.extend(tool_result.result.urls)
                            result_str += f"\n\nSources:\n{'\n'.join(tool_result.result.urls)}"
                        if tool_result.result.image_base64:
                            attachments.append(tool_result.result.image_base64)
                        result_str += (
                            "\n\nDO NOT search again. Write your response NOW using these results."
                        )
                    else:
                        result_str = str(tool_result.result)
                    logger.debug("Tool result: %s", result_str[:200])

                    messages.append(
                        ChatMessage(role=MessageRole.TOOL, content=result_str).to_dict()
                    )

                continue

            # No tool calls - final answer
            content = response.content.strip()

            if not content:
                logger.error("Model returned empty content!")
                return ControllerResponse(answer="Sorry, the model generated an empty response.")

            thinking = response.thinking or response.message.thinking

            if thinking:
                logger.info("Extracted thinking text (length: %d)", len(thinking))

            if source_urls and "http" not in content:
                content += "\n\n" + source_urls[0]

            logger.info("Got final answer (length: %d)", len(content))
            return ControllerResponse(answer=content, thinking=thinking, attachments=attachments)

        logger.warning("Max steps reached without final answer")
        return ControllerResponse(
            answer="Sorry, I couldn't complete that request within the allowed steps."
        )

    async def close(self) -> None:
        """Clean up this agent's resources."""
        await self._ollama_client.close()
        if self in Agent._instances:
            Agent._instances.remove(self)

    @classmethod
    async def close_all(cls) -> None:
        """Close all agent instances."""
        for agent in cls._instances[:]:
            await agent.close()
