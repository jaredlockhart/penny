"""Base Agent class with agentic loop."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole, ToolCallRecord
from penny.config import Config
from penny.database import Database
from penny.ollama import OllamaClient
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool, ToolCall, ToolExecutor, ToolRegistry
from penny.tools.image_search import search_image
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

    Agents receive shared OllamaClient instances — foreground (fast, user-facing)
    and background (smart, processing). Callers create and own the clients;
    agents just hold references.
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
        background_model_client: OllamaClient,
        foreground_model_client: OllamaClient,
        tools: list[Tool],
        db: Database,
        config: Config,
        max_steps: int = 5,
        tool_timeout: float = 60.0,
        vision_model_client: OllamaClient | None = None,
        embedding_model_client: OllamaClient | None = None,
        allow_repeat_tools: bool = False,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.tools = tools
        self.db = db
        self.max_steps = max_steps
        self.allow_repeat_tools = allow_repeat_tools

        self._background_model_client = background_model_client
        self._foreground_model_client = foreground_model_client
        self._vision_model_client = vision_model_client
        self._embedding_model_client = embedding_model_client

        self._tool_registry = ToolRegistry()
        for tool in self.tools:
            self._tool_registry.register(tool)

        self._tool_executor = ToolExecutor(self._tool_registry, timeout=tool_timeout)

        Agent._instances.append(self)

        logger.info(
            "Initialized agent: model=%s, tools=%d, max_steps=%d",
            self._background_model_client.model,
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

    async def _compose_user_facing(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        image_query: str | None = None,
    ) -> ControllerResponse:
        """Compose a user-facing message with system prompt for consistent tone.

        This is the shared primitive for all user-facing model calls.
        Builds messages with the identity prompt and timestamp, calls the model,
        and returns a ControllerResponse with optional image attachment.

        Used directly by proactive notifications (learn agent, extraction),
        and by run() for the no-tool path (e.g. image messages).
        """
        messages = self._build_messages(prompt, history, system_prompt)

        # Run model call and image search concurrently
        try:
            if image_query:
                response, image = await asyncio.gather(
                    self._foreground_model_client.chat(messages=messages),
                    search_image(
                        image_query,
                        api_key=self.config.serper_api_key,
                        max_results=int(self.config.runtime.IMAGE_MAX_RESULTS),
                        timeout=self.config.runtime.IMAGE_DOWNLOAD_TIMEOUT,
                    ),
                )
            else:
                response = await self._foreground_model_client.chat(messages=messages)
                image = None
        except Exception as e:
            logger.error("Failed to compose user-facing message: %s", e)
            return ControllerResponse(answer="")

        content = response.content.strip()
        thinking = response.thinking or response.message.thinking
        attachments = [image] if image else []
        return ControllerResponse(answer=content, thinking=thinking, attachments=attachments)

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
        assert self._vision_model_client is not None
        response = await self._vision_model_client.chat(messages=messages)
        return response.content.strip()

    async def run(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        use_tools: bool = True,
        max_steps: int | None = None,
        system_prompt: str | None = None,
    ) -> ControllerResponse:
        """Run the agent — summary method.

        No-tool path delegates to _compose_user_facing.
        Tool path runs the full agentic loop.
        """
        tools = self._tool_registry.get_ollama_tools() if use_tools else None
        logger.debug("Using %d tools", len(tools) if tools else 0)

        if not tools:
            return await self._compose_user_facing(prompt, history, system_prompt)

        messages = self._build_messages(prompt, history, system_prompt)
        steps = max_steps if max_steps is not None else self.max_steps
        return await self._run_agentic_loop(messages, tools, steps)

    async def _run_agentic_loop(
        self,
        messages: list[dict],
        tools: list[dict],
        steps: int,
    ) -> ControllerResponse:
        """Execute the step loop: call model, process tool calls, or return final answer."""
        attachments: list[str] = []
        source_urls: list[str] = []
        called_tools: set[str] = set()
        tool_call_records: list[ToolCallRecord] = []

        for step in range(steps):
            logger.info("Agent step %d/%d", step + 1, steps)

            response = await self._call_model_with_xml_retry(messages, tools)
            if response is None:
                return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)

            if response.has_tool_calls:
                await self._process_tool_calls(
                    response, messages, called_tools, tool_call_records, source_urls, attachments
                )
                continue

            return self._build_final_response(response, source_urls, attachments, tool_call_records)

        logger.warning("Max steps reached without final answer")
        return ControllerResponse(
            answer=PennyResponse.AGENT_MAX_STEPS, tool_calls=tool_call_records
        )

    async def _call_model_with_xml_retry(self, messages: list[dict], tools: list[dict]):
        """Call the model, retrying if it emits XML markup instead of structured tool calls."""
        max_xml_retries = 3
        response = None

        for xml_attempt in range(max_xml_retries):
            try:
                response = await self._background_model_client.chat(messages=messages, tools=tools)
            except Exception as e:
                logger.error("Error calling Ollama: %s", e)
                return None

            if response.has_tool_calls:
                break

            content = response.content.strip()
            if not _has_xml_tags(content):
                break

            logger.warning(
                "Model emitted XML markup in content; retrying (attempt %d/%d)",
                xml_attempt + 1,
                max_xml_retries,
            )

        return response

    async def _process_tool_calls(
        self,
        response,
        messages: list[dict],
        called_tools: set[str],
        tool_call_records: list[ToolCallRecord],
        source_urls: list[str],
        attachments: list[str],
    ) -> None:
        """Handle all tool calls from a single model response."""
        logger.info("Model requested %d tool call(s)", len(response.message.tool_calls or []))
        messages.append(response.message.to_input_message())

        for ollama_tool_call in response.message.tool_calls or []:
            tool_name = ollama_tool_call.function.name
            arguments = ollama_tool_call.function.arguments

            if not self.allow_repeat_tools and tool_name in called_tools:
                logger.info("Skipping repeat call to tool: %s", tool_name)
                repeat_msg = "Tool already called. DO NOT search again. Write your response NOW."
                messages.append(ChatMessage(role=MessageRole.TOOL, content=repeat_msg).to_dict())
                continue

            result_str = await self._execute_single_tool(
                tool_name, arguments, called_tools, tool_call_records, source_urls, attachments
            )
            messages.append(ChatMessage(role=MessageRole.TOOL, content=result_str).to_dict())

    async def _execute_single_tool(
        self,
        tool_name: str,
        arguments: dict,
        called_tools: set[str],
        tool_call_records: list[ToolCallRecord],
        source_urls: list[str],
        attachments: list[str],
    ) -> str:
        """Execute one tool call, update tracking state, return result string."""
        logger.info("Executing tool: %s", tool_name)
        called_tools.add(tool_name)
        tool_call_records.append(ToolCallRecord(tool=tool_name, arguments=arguments))

        tool_call = ToolCall(tool=tool_name, arguments=arguments)
        tool_result = await self._tool_executor.execute(tool_call)

        if tool_result.error:
            result_str = f"Error: {tool_result.error}"
        elif isinstance(tool_result.result, SearchResult):
            result_str = self._format_search_result(tool_result.result, source_urls, attachments)
        else:
            result_str = str(tool_result.result)

        logger.debug("Tool result: %s", result_str[:200])
        return result_str

    def _format_search_result(
        self, result: SearchResult, source_urls: list[str], attachments: list[str]
    ) -> str:
        """Format a SearchResult into a string, collecting URLs and images."""
        text = result.text
        if result.urls:
            source_urls.extend(result.urls)
            text += f"\n\nSources:\n{'\n'.join(result.urls)}"
        if result.image_base64:
            attachments.append(result.image_base64)
        text += "\n\nDO NOT search again. Write your response NOW using these results."
        return text

    def _build_final_response(
        self,
        response,
        source_urls: list[str],
        attachments: list[str],
        tool_call_records: list[ToolCallRecord],
    ) -> ControllerResponse:
        """Build the ControllerResponse from the model's final (non-tool) answer."""
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

    async def close(self) -> None:
        """Remove this agent from the instance registry."""
        if self in Agent._instances:
            Agent._instances.remove(self)

    @classmethod
    async def close_all(cls) -> None:
        """Close all agent instances."""
        for agent in cls._instances[:]:
            await agent.close()
