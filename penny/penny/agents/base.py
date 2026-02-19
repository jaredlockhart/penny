"""Base Agent class with agentic loop."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole, ToolCallRecord
from penny.database import Database
from penny.ollama import OllamaClient
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool, ToolCall, ToolExecutor, ToolRegistry
from penny.tools.models import SearchResult

logger = logging.getLogger(__name__)


# Matches paired XML-like tags in content, e.g. <function=search>...</function>
# or <tools><search>...</search></tools>
_XML_TAG_PATTERN = re.compile(r"<[a-zA-Z]\w*[\s=>].*</[a-zA-Z]\w*>", re.DOTALL)


def _has_xml_tags(content: str) -> bool:
    """Return True if content contains XML-like tag pairs."""
    return bool(_XML_TAG_PATTERN.search(content))


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
        embedding_model: str | None = None,
        allow_repeat_tools: bool = False,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools
        self.db = db
        self.max_steps = max_steps
        self.vision_model = vision_model
        self.embedding_model = embedding_model
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
    ) -> list[dict]:
        """Build message list for Ollama chat API.

        Args:
            prompt: The user message/prompt to respond to
            history: Optional conversation history as (role, content) tuples
            system_prompt: Optional system prompt override

        Returns:
            List of message dicts for Ollama chat API
        """
        messages = []

        effective_prompt = system_prompt or self.system_prompt
        now = datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M %p UTC")

        # Build system prompt: timestamp → identity → agent-specific prompt
        system_parts = [f"Current date and time: {now}", ""]

        system_parts.append(Prompt.PENNY_IDENTITY)

        if effective_prompt:
            system_parts.append("")
            system_parts.append(effective_prompt)

        system_content = "\n".join(system_parts)
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content).to_dict())

        if history:
            for role, content in history:
                messages.append(ChatMessage(role=MessageRole(role), content=content).to_dict())

        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)
        messages.append(user_msg.to_dict())

        return messages

    async def _compose_user_facing(self, prompt: str) -> str:
        """Compose a user-facing message with system prompt for consistent tone.

        Uses _build_messages to include the identity prompt and timestamp,
        ensuring proactive messages have the same context as normal responses.

        Returns the model's response text, or empty string on failure.
        """
        messages = self._build_messages(prompt)
        try:
            response = await self._ollama_client.chat(messages=messages)
            return response.content.strip()
        except Exception as e:
            logger.error("Failed to compose user-facing message: %s", e)
            return ""

    async def caption_image(self, image_b64: str) -> str:
        """Caption an image using the vision model.

        Args:
            image_b64: Base64-encoded image data

        Returns:
            Text description of the image
        """
        messages = [
            {"role": "user", "content": Prompt.VISION_AUTO_DESCRIBE_PROMPT, "images": [image_b64]},
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
    ) -> ControllerResponse:
        """
        Run the agent with a prompt.

        Args:
            prompt: The user message/prompt to respond to
            history: Optional conversation history as (role, content) tuples
            use_tools: Whether to enable tools for this run (default: True)
            max_steps: Override max_steps for this run (default: use agent's max_steps)
            system_prompt: Override system prompt for this run (default: use agent's prompt)

        Returns:
            ControllerResponse with answer, thinking, and attachments
        """
        messages = self._build_messages(prompt, history, system_prompt)
        tools = self._tool_registry.get_ollama_tools() if use_tools else None
        logger.debug("Using %d tools", len(tools) if tools else 0)

        attachments: list[str] = []
        source_urls: list[str] = []
        called_tools: set[str] = set()
        tool_call_records: list[ToolCallRecord] = []

        max_xml_retries = 3
        steps = max_steps if max_steps is not None else self.max_steps
        for step in range(steps):
            logger.info("Agent step %d/%d", step + 1, steps)

            # Retry the model call if it emits XML markup instead of using
            # structured tool_calls. This doesn't consume an agentic loop step.
            for xml_attempt in range(max_xml_retries):
                try:
                    response = await self._ollama_client.chat(messages=messages, tools=tools)
                except Exception as e:
                    logger.error("Error calling Ollama: %s", e)
                    return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)

                if response.has_tool_calls:
                    break

                content = response.content.strip()
                if not (tools and _has_xml_tags(content)):
                    break

                logger.warning(
                    "Model emitted XML markup in content; retrying (attempt %d/%d)",
                    xml_attempt + 1,
                    max_xml_retries,
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
                    tool_call_records.append(ToolCallRecord(tool=tool_name, arguments=arguments))

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

            # No tool calls — final answer
            content = response.content.strip()

            if not content:
                logger.error("Model returned empty content!")
                return ControllerResponse(answer=PennyResponse.AGENT_EMPTY_RESPONSE)

            thinking = response.thinking or response.message.thinking

            if thinking:
                logger.info("Extracted thinking text (length: %d)", len(thinking))

            if source_urls and "http" not in content:
                content += "\n\n" + source_urls[0]

            logger.info("Got final answer (length: %d)", len(content))
            return ControllerResponse(
                answer=content,
                thinking=thinking,
                attachments=attachments,
                tool_calls=tool_call_records,
            )

        logger.warning("Max steps reached without final answer")
        return ControllerResponse(
            answer=PennyResponse.AGENT_MAX_STEPS,
            tool_calls=tool_call_records,
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
