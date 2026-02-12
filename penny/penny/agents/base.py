"""Base Agent class with agentic loop."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole
from penny.constants import VISION_AUTO_DESCRIBE_PROMPT
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
        vision_model: str | None = None,
        allow_repeat_tools: bool = False,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools
        self.db = db
        self.max_steps = max_steps
        self.vision_model = vision_model
        self.allow_repeat_tools = allow_repeat_tools

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
        system_prompt: str | None = None,
        sender: str | None = None,
    ) -> list[dict]:
        """Build message list for Ollama chat API.

        Args:
            prompt: The user message/prompt to respond to
            history: Optional conversation history as (role, content) tuples
            system_prompt: Optional system prompt override
            sender: Optional user identifier for personality injection

        Returns:
            List of message dicts for Ollama chat API
        """
        messages = []

        effective_prompt = system_prompt or self.system_prompt
        now = datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M %p UTC")

        # Build system prompt with personality injection
        system_parts = [f"Current date and time: {now}", ""]

        # Check if user has custom personality prompt
        personality_text = None
        if sender:
            try:
                personality = self.db.get_personality_prompt(sender)
                if personality:
                    personality_text = personality.prompt_text
            except Exception as e:
                # Table might not exist yet (e.g., during migration or in test snapshots)
                logger.debug("Failed to query personality prompt: %s", e)
                personality_text = None

        # Inject personality between base identity and agent-specific prompt
        # The system_prompt already includes PENNY_IDENTITY, so we need to reconstruct it
        from penny.constants import PENNY_IDENTITY

        # Start with base Penny identity
        system_parts.append(PENNY_IDENTITY)

        # Add custom personality if it exists
        if personality_text:
            system_parts.append("")
            system_parts.append(personality_text)

        # Add agent-specific prompt (remove PENNY_IDENTITY if present)
        agent_prompt = effective_prompt
        if agent_prompt.startswith(PENNY_IDENTITY):
            # Remove the base identity since we already added it
            agent_prompt = agent_prompt[len(PENNY_IDENTITY) :].lstrip("\n")

        if agent_prompt:
            system_parts.append("")
            system_parts.append(agent_prompt)

        system_content = "\n".join(system_parts)
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content).to_dict())

        if history:
            for role, content in history:
                messages.append(ChatMessage(role=MessageRole(role), content=content).to_dict())

        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)
        messages.append(user_msg.to_dict())

        return messages

    async def caption_image(self, image_b64: str) -> str:
        """Caption an image using the vision model.

        Args:
            image_b64: Base64-encoded image data

        Returns:
            Text description of the image
        """
        messages = [
            {"role": "user", "content": VISION_AUTO_DESCRIBE_PROMPT, "images": [image_b64]},
        ]
        response = await self._ollama_client.chat(messages=messages, model=self.vision_model)
        return response.content.strip()

    async def run(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        use_tools: bool = True,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        sender: str | None = None,
    ) -> ControllerResponse:
        """
        Run the agent with a prompt.

        Args:
            prompt: The user message/prompt to respond to
            history: Optional conversation history as (role, content) tuples
            use_tools: Whether to enable tools for this run (default: True)
            max_steps: Override max_steps for this run (default: use agent's max_steps)
            system_prompt: Override system prompt for this run (default: use agent's prompt)
            sender: Optional user identifier for personality injection

        Returns:
            ControllerResponse with answer, thinking, and attachments
        """
        messages = self._build_messages(prompt, history, system_prompt, sender)
        tools = self._tool_registry.get_ollama_tools() if use_tools else None
        logger.debug("Using %d tools", len(tools) if tools else 0)

        attachments: list[str] = []
        source_urls: list[str] = []
        called_tools: set[str] = set()

        steps = max_steps if max_steps is not None else self.max_steps
        for step in range(steps):
            logger.info("Agent step %d/%d", step + 1, steps)

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

                    if not self.allow_repeat_tools and tool_name in called_tools:
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
