"""Base Agent class with agentic loop and context building."""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse as _urlparse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole, ToolCallRecord
from penny.config import Config
from penny.constants import PennyConstants
from penny.database import Database
from penny.ollama import OllamaClient
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool, ToolCall, ToolExecutor, ToolRegistry
from penny.tools.models import SearchResult
from penny.tools.multi import MultiTool

if TYPE_CHECKING:
    from penny.tools.browse_url import BrowseUrlTool

logger = logging.getLogger(__name__)


# Matches paired XML-like tags in content, e.g. <function=search>...</function>
# or <tools><search>...</search></tools>
_XML_TAG_PATTERN = re.compile(r"<[a-zA-Z]\w*[\s=>].*</[a-zA-Z]\w*>", re.DOTALL)

# Matches <think>...</think> blocks emitted inline by some models (e.g. DeepSeek-R1, Qwen3)
_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Matches markdown links [text](url) and bare URLs for validation
_MARKDOWN_LINK_URL_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")
_BARE_URL_PATTERN = re.compile(r"(?<!\()(https?://\S+)")


def _is_url_truncated(url: str) -> bool:
    """Return True if url appears truncated or malformed.

    Checks for missing host and trailing hyphen (the most common sign of a cut-off path).
    Strips trailing prose punctuation before validation so sentence-ending periods
    don't cause false positives.
    """
    cleaned = url.rstrip(".,;:!?\"')>}]")
    try:
        parsed = _urlparse.urlparse(cleaned)
    except Exception:
        return True
    if not parsed.netloc or "." not in parsed.netloc:
        return True
    return cleaned.endswith("-")


def _clean_malformed_urls(content: str) -> str:
    """Remove truncated or malformed URLs from model-generated content.

    For markdown links [text](bad_url), the link text is preserved.
    For bare malformed URLs, the URL token is removed entirely.
    Valid URLs are left unchanged.
    """

    def fix_md_link(m: re.Match) -> str:
        text, url = m.group(1), m.group(2)
        if _is_url_truncated(url):
            logger.warning("Stripped malformed URL from markdown link: %.120s", url)
            return text
        return m.group(0)

    def fix_bare_url(m: re.Match) -> str:
        url = m.group(1)
        if _is_url_truncated(url):
            logger.warning("Stripped malformed bare URL: %.120s", url)
            return ""
        return m.group(0)

    content = _MARKDOWN_LINK_URL_PATTERN.sub(fix_md_link, content)
    content = _BARE_URL_PATTERN.sub(fix_bare_url, content)
    return content


def _has_xml_tags(content: str) -> bool:
    """Return True if content contains XML-like tag pairs."""
    return bool(_XML_TAG_PATTERN.search(content))


def _strip_think_tags(content: str) -> tuple[str, str | None]:
    """Strip <think>...</think> blocks from content.

    Returns (cleaned_content, extracted_thinking) where extracted_thinking
    contains the concatenated text from all stripped blocks.
    """
    thinking_parts: list[str] = []

    def _collect(m: re.Match) -> str:
        thinking_parts.append(m.group(1).strip())
        return ""

    cleaned = _THINK_TAG_PATTERN.sub(_collect, content).strip()
    extracted = "\n\n".join(thinking_parts) if thinking_parts else None
    return cleaned, extracted


# Prefixes in tool result strings that indicate a failed or empty result.
# Checked after tool execution to mark ToolCallRecord.failed = True.
_TOOL_FAILURE_PREFIXES = (
    "Error: ",
    PennyResponse.NO_RESULTS_TEXT,
    "Failed to search:",
    "No browser connected",
)


def _is_tool_result_failed(result_str: str) -> bool:
    """Return True if a tool result indicates failure (error, no results, quota exceeded)."""
    return any(result_str.startswith(prefix) for prefix in _TOOL_FAILURE_PREFIXES)


_TOOL_RESULT_TRUNCATION_THRESHOLD = PennyConstants.TOOL_RESULT_TRUNCATION_THRESHOLD
_TOOL_RESULT_MAX_CHARS = PennyConstants.TOOL_RESULT_TRUNCATION_MAX_CHARS


def _build_strong_nudge(messages: list[dict]) -> str:
    """Build a context-aware nudge that includes the original user question.

    Called when many preceding tool calls may have saturated the model's context.
    Including the original question gives the model a clear target after heavy tool use.
    """
    user_messages = [
        m["content"]
        for m in messages
        if m.get("role") == MessageRole.USER and not m["content"].startswith("You have gathered")
    ]
    original_question = user_messages[-1] if user_messages else None
    if original_question:
        return (
            f"You have gathered enough information from your searches. "
            f"Please provide your final answer to: {original_question}"
        )
    return "You have gathered enough information. Please provide your final response."


# Phrases that indicate a model refusal — used to detect and retry unhelpful responses
_REFUSAL_PHRASES = (
    "i can't",
    "i cannot",
    "i'm sorry",
    "i am sorry",
    "i'm unable",
    "i am unable",
    "i apologize",
    "as an ai",
    "as a language model",
)


@dataclass
class _StepResult:
    """Result of processing all tool calls in one agentic loop step."""

    messages: list[dict]
    records: list[ToolCallRecord]
    source_urls: list[str]
    attachments: list[str]


class Agent:
    """
    AI agent with a specific persona and capabilities.

    Agents receive shared OllamaClient instances — foreground (fast, user-facing)
    and background (smart, processing). Callers create and own the clients;
    agents just hold references.
    """

    _instances: list[Agent] = []

    THOUGHT_CONTEXT_LIMIT = PennyConstants.THOUGHT_CONTEXT_LIMIT
    PREFERRED_POOL_SIZE = PennyConstants.PREFERRED_POOL_SIZE
    MAX_TOOL_RESULT_CHARS = PennyConstants.MAX_TOOL_RESULT_CHARS
    name: str = "Agent"

    def __init__(
        self,
        system_prompt: str,
        model_client: OllamaClient,
        tools: list[Tool],
        db: Database,
        config: Config,
        tool_timeout: float = 60.0,
        vision_model_client: OllamaClient | None = None,
        embedding_model_client: OllamaClient | None = None,
        allow_repeat_tools: bool = False,
        max_queries_key: str | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.tools = tools
        self.db = db
        self.allow_repeat_tools = allow_repeat_tools

        self._model_client = model_client
        self._vision_model_client = vision_model_client
        self._embedding_model_client = embedding_model_client

        self._max_queries_key = max_queries_key
        self._multi_tool: MultiTool | None = None
        self._browse_url_provider: Callable[[], BrowseUrlTool | None] | None = None
        self._browser_tools_provider: Callable[[], list[Tool]] | None = None
        self._current_user: str | None = None
        self._tool_result_text: list[str] = []

        self._tool_registry = ToolRegistry()
        for tool in self.tools:
            self._tool_registry.register(tool)

        self._tool_executor = ToolExecutor(self._tool_registry, timeout=tool_timeout)
        self._keep_tools_on_final_step = False
        self._on_tool_start_factory: (
            Callable[
                [],
                tuple[
                    Callable[[list[tuple[str, dict]]], Awaitable[None]],
                    Callable[[], Awaitable[None]],
                ],
            ]
            | None
        ) = None

        Agent._instances.append(self)

        logger.info(
            "Initialized agent: model=%s, tools=%d",
            self._model_client.model,
            len(self.tools),
        )

    # ── Top-level execution ──────────────────────────────────────────────

    async def execute(self) -> bool:
        """Run a scheduled cycle — iterate users and delegate to execute_for_user.

        Override execute_for_user for per-user work, or override execute
        entirely for non-user-based work.
        """
        users = self.get_users()
        if not users:
            return False
        did_work = False
        for user in users:
            if await self.execute_for_user(user):
                did_work = True
        return did_work

    async def execute_for_user(self, user: str) -> bool:
        """Standard scheduled cycle: build tools, get prompt, run loop, post-process."""
        self._current_user = user
        try:
            tools = self.get_tools(user)
            if not tools:
                return False
            self._install_tools(tools)

            prompt = await self.get_prompt(user)
            if not prompt:
                return False

            logger.info("%s starting for %s", self.name, user)
            system_prompt = await self._build_system_prompt(user)
            history = self.get_history(user)
            on_tool_start, tool_cleanup = (
                self._on_tool_start_factory() if self._on_tool_start_factory else (None, None)
            )
            try:
                await self.run(
                    prompt=prompt,
                    max_steps=self.get_max_steps(),
                    system_prompt=system_prompt,
                    history=history,
                    on_tool_start=on_tool_start,
                )
            finally:
                if tool_cleanup:
                    await tool_cleanup()
            did_work = await self.after_run(user)
            logger.info("%s complete for %s", self.name, user)
            return did_work
        except Exception:
            logger.exception("%s failed for %s", self.name, user)
            return False
        finally:
            self._current_user = None

    # ── Override hooks ───────────────────────────────────────────────────

    def get_max_steps(self) -> int:
        """Return max agentic loop steps. Subclasses must override."""
        raise NotImplementedError(f"{type(self).__name__} must override get_max_steps()")

    def get_users(self) -> list[str]:
        """Return the single user to process (Penny is single-user)."""
        primary = self.db.users.get_primary_sender()
        return [primary] if primary else []

    async def get_prompt(self, user: str) -> str | None:
        """Build the prompt for the agentic loop. Return None to skip this user."""
        return None

    def get_history(self, user: str) -> list[tuple[str, str]] | None:
        """Conversation history for the agentic loop. Override for conversation agents."""
        return None

    async def after_run(self, user: str) -> bool:
        """Post-processing after the agentic loop. Return True if work was done."""
        return True

    # ── Agentic loop entry ───────────────────────────────────────────────

    async def run(
        self,
        prompt: str,
        max_steps: int,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
    ) -> ControllerResponse:
        """Run the agentic loop — prompt in, response out."""
        self._tool_result_text = []
        messages = self._build_messages(prompt, history, system_prompt)
        tools = self._tool_registry.get_ollama_tools()
        return await self._run_agentic_loop(messages, tools, max_steps, on_tool_start)

    # ── Agentic loop internals ───────────────────────────────────────────

    @staticmethod
    def _is_refusal(content: str) -> bool:
        """Return True if content looks like a model refusal."""
        lower = content.lower()
        return any(phrase in lower for phrase in _REFUSAL_PHRASES)

    async def _run_agentic_loop(
        self,
        messages: list[dict],
        tools: list[dict],
        steps: int,
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
    ) -> ControllerResponse:
        """Execute the step loop: call model, process tool calls, or return final answer."""
        attachments: list[str] = []
        source_urls: list[str] = []
        called_tools: set[tuple[str, ...]] = set()
        tool_call_records: list[ToolCallRecord] = []
        empty_retries: int = 0
        refusal_retries: int = 0

        force_no_tools = False

        for step in range(steps):
            logger.info("Agent step %d/%d", step + 1, steps)

            is_final_step = step == steps - 1
            strip_tools = force_no_tools or (is_final_step and not self._keep_tools_on_final_step)
            step_tools = [] if strip_tools else tools
            if strip_tools:
                logger.debug("Final step — tools removed, model must produce text")

            response = await self._call_model_with_xml_retry(messages, step_tools)
            if response is None:
                return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)

            self.on_response(response)

            if response.has_tool_calls and strip_tools:
                logger.warning("Model hallucinated tool calls on final step — ignoring")

            if response.has_tool_calls and not strip_tools:
                result = await self._process_tool_calls(response, called_tools, on_tool_start)
                messages.extend(result.messages)
                tool_call_records.extend(result.records)
                source_urls.extend(result.source_urls)
                attachments.extend(result.attachments)
                await self.after_step(result.records, result.messages)
                if self.should_stop_loop(result.records):
                    logger.info("Loop stop requested after step %d/%d", step + 1, steps)
                    break
                # If every tool call so far has failed and there have been at least
                # two, abort early rather than letting the model hallucinate from
                # nothing. A single failure gets one more chance — the model sees
                # the error and may produce text or try a different query.
                if len(tool_call_records) >= PennyConstants.TOOL_FAILURE_ABORT_THRESHOLD and all(
                    r.failed for r in tool_call_records
                ):
                    failed_tools = sorted({r.tool for r in tool_call_records})
                    logger.warning(
                        "All %d tool call(s) failed — aborting: %s",
                        len(tool_call_records),
                        ", ".join(failed_tools),
                    )
                    return ControllerResponse(
                        answer=PennyResponse.AGENT_TOOLS_UNAVAILABLE.format(
                            tools=", ".join(failed_tools)
                        ),
                        tool_calls=tool_call_records,
                    )
                # Reset empty retry counter so the synthesis step gets a retry even if
                # a previous intermediate step already consumed it.
                empty_retries = 0
                continue

            if await self.handle_text_step(response, messages, step, is_final_step):
                continue

            # Strip think tags before checking emptiness — model may return only
            # <think>...</think> with no body text, which would bypass the empty check.
            effective_content, _ = _strip_think_tags(response.content.strip())
            if not effective_content and empty_retries == 0:
                empty_retries += 1
                logger.warning(
                    "Model returned empty content on step %d/%d; requesting text output",
                    step + 1,
                    steps,
                )
                messages.append(response.message.to_input_message())
                if len(tool_call_records) >= _TOOL_RESULT_TRUNCATION_THRESHOLD:
                    logger.warning(
                        "Truncating tool results before retry (preceding_tool_calls=%d)",
                        len(tool_call_records),
                    )
                    messages = self._truncate_tool_messages(messages)
                    nudge = _build_strong_nudge(messages)
                elif tool_call_records:
                    nudge = (
                        "You've completed your research. Please synthesize your findings "
                        "and provide a helpful response."
                    )
                else:
                    nudge = "Please provide your response."
                messages.append({"role": MessageRole.USER, "content": nudge})
                force_no_tools = True
                if not is_final_step:
                    continue
                # On the final step, retry directly — can't extend a for-range loop
                response = await self._call_model_with_xml_retry(messages, step_tools)
                if response is None:
                    return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)
                self.on_response(response)

            if (
                refusal_retries == 0
                and response.content.strip()
                and self._is_refusal(response.content.strip())
            ):
                refusal_retries += 1
                logger.warning(
                    "Model returned refusal on step %d/%d; nudging for substantive response",
                    step + 1,
                    steps,
                )
                messages.append(response.message.to_input_message())
                messages.append(
                    {
                        "role": MessageRole.USER,
                        "content": (
                            "Please provide a helpful response. "
                            "Use your search tools or what you know to give a useful answer."
                        ),
                    }
                )
                if not is_final_step:
                    continue
                # On the final step, retry directly
                response = await self._call_model_with_xml_retry(messages, step_tools)
                if response is None:
                    return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)
                self.on_response(response)

            return self._build_final_response(response, source_urls, attachments, tool_call_records)

        logger.warning("Max steps reached without final answer")
        return ControllerResponse(
            answer=PennyResponse.AGENT_MAX_STEPS, tool_calls=tool_call_records
        )

    def on_response(self, response) -> None:
        """Hook called after every model response, before tool/text branching.

        Override to capture content from all responses (e.g. inner monologue).
        """

    async def handle_text_step(
        self, response, messages: list[dict], step: int, is_final: bool
    ) -> bool:
        """Handle a text-only model response. Return True to continue, False to stop.

        Base returns False — text response = final answer.
        Override to inject continuation messages and keep the loop going.
        """
        return False

    async def after_step(self, step_records: list[ToolCallRecord], messages: list[dict]) -> None:
        """Capture tool result text for URL validation. Override in subclasses (call super)."""
        for msg in messages:
            if msg.get("role") == MessageRole.TOOL:
                content = msg.get("content", "")
                if content:
                    self._tool_result_text.append(content)

    def should_stop_loop(self, step_records: list[ToolCallRecord]) -> bool:
        """Check if the loop should stop early. Override in subclasses."""
        return False

    async def _call_model_with_xml_retry(self, messages: list[dict], tools: list[dict]):
        """Call the model, retrying if it emits XML markup instead of structured tool calls."""
        max_xml_retries = PennyConstants.XML_RETRY_LIMIT
        response = None
        effective_tools = tools if tools else None

        for xml_attempt in range(max_xml_retries):
            try:
                response = await self._model_client.chat(messages=messages, tools=effective_tools)
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

    @staticmethod
    def _truncate_tool_messages(messages: list[dict]) -> list[dict]:
        """Truncate tool result messages to reduce context size before retrying.

        Called when many preceding tool calls may have saturated the model's
        context window, causing an empty response on the synthesis step.
        """
        result = []
        for msg in messages:
            if msg.get("role") == MessageRole.TOOL:
                content = msg.get("content", "")
                if len(content) > _TOOL_RESULT_MAX_CHARS:
                    msg = {**msg, "content": content[:_TOOL_RESULT_MAX_CHARS] + "... [truncated]"}
            result.append(msg)
        return result

    def _build_final_response(
        self,
        response,
        source_urls: list[str],
        attachments: list[str],
        tool_call_records: list[ToolCallRecord],
    ) -> ControllerResponse:
        """Build the ControllerResponse from the model's final (non-tool) answer."""
        logger.debug("Building final response with %d attachments", len(attachments))
        content = response.content.strip()

        if not content:
            logger.error(
                "Model returned empty content! model=%s, preceding_tool_calls=%d",
                self._model_client.model,
                len(tool_call_records),
            )
            fallback = (
                PennyResponse.FALLBACK_RESPONSE
                if tool_call_records
                else PennyResponse.AGENT_EMPTY_RESPONSE
            )
            return ControllerResponse(answer=fallback)

        thinking = response.thinking or response.message.thinking

        # Strip <think>...</think> blocks emitted inline by some models.
        # Move extracted content to the thinking field if not already populated.
        content, inline_thinking = _strip_think_tags(content)
        if not thinking and inline_thinking:
            thinking = inline_thinking

        if thinking:
            logger.info("Extracted thinking text (length: %d)", len(thinking))

        if not content:
            logger.error("Model returned empty content after stripping think tags!")
            fallback = (
                PennyResponse.FALLBACK_RESPONSE
                if tool_call_records
                else PennyResponse.AGENT_EMPTY_RESPONSE
            )
            return ControllerResponse(answer=fallback)

        content = _clean_malformed_urls(content)

        if source_urls and "http" not in content:
            content += "\n\n" + source_urls[0]

        word_count = len(content.split())
        if word_count < 10:
            logger.warning("Short response detected (word_count=%d): %s", word_count, content[:100])
        logger.info("Got final answer (length: %d)", len(content))
        return ControllerResponse(
            answer=content,
            thinking=thinking,
            attachments=attachments,
            tool_calls=tool_call_records,
        )

    # ── Tool management ──────────────────────────────────────────────────

    def set_browser_tools_provider(self, provider: Callable[[], list[Tool]]) -> None:
        """Set a callback that provides browser tools when a browser is connected."""
        self._browser_tools_provider = provider

    def get_tools(self, user: str) -> list[Tool]:
        """Build tool list for this agent.

        When max_queries_key is set, builds a fresh MultiTool each cycle
        so runtime config changes take effect immediately.
        """
        if self._max_queries_key is not None:
            return [self._build_multi_tool()]
        tools: list[Tool] = []
        if self._browser_tools_provider:
            tools.extend(self._browser_tools_provider())
        return tools

    def _build_multi_tool(self) -> MultiTool:
        """Build a fresh MultiTool from config, updating self._multi_tool."""
        assert self._max_queries_key is not None
        max_calls = int(getattr(self.config.runtime, self._max_queries_key))
        tool = MultiTool(max_calls=max_calls)
        if self._browse_url_provider:
            tool.set_browse_url_provider(self._browse_url_provider)
        self._multi_tool = tool
        return tool

    def _install_tools(self, tools: list[Tool]) -> None:
        """Replace the agent's tool registry and executor."""
        self._tool_registry = ToolRegistry()
        for tool in tools:
            self._tool_registry.register(tool)
        self._tool_executor = ToolExecutor(self._tool_registry, timeout=self.config.tool_timeout)

    async def _process_tool_calls(
        self,
        response,
        called_tools: set[tuple[str, ...]],
        on_tool_start: Callable[[list[tuple[str, dict]]], Awaitable[None]] | None = None,
    ) -> _StepResult:
        """Process all tool calls from a model response, executing valid ones in parallel."""
        logger.info("Model requested %d tool call(s)", len(response.message.tool_calls or []))
        messages: list[dict] = [response.message.to_input_message()]
        records: list[ToolCallRecord] = []
        source_urls: list[str] = []
        attachments: list[str] = []

        # Dedup check and on_tool_start are sequential: dedup requires ordered mutation of
        # called_tools, and on_tool_start fires UI status updates before execution begins.
        max_calls = int(self.config.runtime.MESSAGE_MAX_TOOL_CALLS) if self.config else 5
        pending: list[tuple[str, dict, str | None]] = []
        for ollama_tool_call in (response.message.tool_calls or [])[:max_calls]:
            tool_name = ollama_tool_call.function.name
            arguments = ollama_tool_call.function.arguments

            # Pop reasoning before dedup (same args + different reasoning = repeat)
            reasoning = arguments.pop("reasoning", None)
            call_key = self._make_call_key(tool_name, arguments)

            if not self.allow_repeat_tools and call_key in called_tools:
                logger.info("Skipping repeat: %s(%s)", tool_name, arguments)
                repeat_msg = "You already made this exact tool call. Try a different query or tool."
                messages.append(
                    {"role": MessageRole.TOOL, "content": repeat_msg, "tool_name": tool_name}
                )
                continue

            called_tools.add(call_key)
            pending.append((tool_name, arguments, reasoning))

        # Fire on_tool_start once with all pending tools so the UI can show
        # a combined status (e.g. "Searching A + Searching B") for parallel calls.
        if on_tool_start and pending:
            try:
                await on_tool_start([(name, dict(args)) for name, args, _ in pending])
            except Exception:
                logger.debug("on_tool_start callback failed")

        # Execute all valid tool calls in parallel.
        results = await asyncio.gather(
            *[self._execute_single_tool(name, args, reasoning) for name, args, reasoning in pending]
        )

        for (tool_name, _, _), (result_str, record, urls, image) in zip(
            pending, results, strict=True
        ):
            records.append(record)
            source_urls.extend(urls)
            if image:
                attachments.append(image)
            messages.append(
                {"role": MessageRole.TOOL, "content": result_str, "tool_name": tool_name}
            )

        return _StepResult(
            messages=messages,
            records=records,
            source_urls=source_urls,
            attachments=attachments,
        )

    async def _execute_single_tool(
        self,
        tool_name: str,
        arguments: dict,
        reasoning: str | None,
    ) -> tuple[str, ToolCallRecord, list[str], str | None]:
        """Execute one tool call. Returns (result_str, record, source_urls, image)."""
        logger.info("Executing tool: %s", tool_name)
        if reasoning:
            logger.debug("Tool reasoning: %s", reasoning[:200])

        record = ToolCallRecord(tool=tool_name, arguments=arguments, reasoning=reasoning)
        tool_call = ToolCall(tool=tool_name, arguments=arguments)
        tool_result = await self._tool_executor.execute(tool_call)

        if tool_result.error:
            result_str = f"Error: {tool_result.error}"
            record.failed = True
            logger.debug("Tool result (failed): %s", result_str[:200])
            return result_str, record, [], None

        if isinstance(tool_result.result, SearchResult):
            result_str, urls, image = self._format_search_result(tool_result.result)
            result_str = self._truncate_tool_result(result_str)
            record.failed = _is_tool_result_failed(result_str)
            logger.debug("Tool result: %s", result_str[:200])
            return result_str, record, urls, image
        result_str = self._truncate_tool_result(str(tool_result.result))
        record.failed = _is_tool_result_failed(result_str)
        logger.debug("Tool result: %s", result_str[:200])
        return result_str, record, [], None

    def _truncate_tool_result(self, result_str: str) -> str:
        """Truncate tool result to MAX_TOOL_RESULT_CHARS to prevent context saturation."""
        if len(result_str) <= self.MAX_TOOL_RESULT_CHARS:
            return result_str
        logger.warning(
            "Tool result truncated from %d to %d chars to prevent context saturation",
            len(result_str),
            self.MAX_TOOL_RESULT_CHARS,
        )
        return result_str[: self.MAX_TOOL_RESULT_CHARS] + " [truncated]"

    @staticmethod
    def _make_call_key(tool_name: str, arguments: dict) -> tuple[str, ...]:
        """Build a hashable key from tool name + arguments for dedup."""
        arg_parts = tuple(f"{k}={v}" for k, v in sorted(arguments.items()))
        return (tool_name, *arg_parts)

    @staticmethod
    def _format_search_result(result: SearchResult) -> tuple[str, list[str], str | None]:
        """Format a SearchResult. Returns (text, urls, image_base64)."""
        text = result.text
        urls = result.urls or []
        if urls:
            sources = "\n".join(urls)
            text += f"\n\nSources:\n{sources}"
        return text, urls, result.image_base64

    # ── URL validation ──────────────────────────────────────────────────

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        """Extract all URLs from text (both markdown links and bare URLs)."""
        md_urls = [m.group(2) for m in _MARKDOWN_LINK_URL_PATTERN.finditer(text)]
        bare_urls = [m.group(1) for m in _BARE_URL_PATTERN.finditer(text)]
        seen: set[str] = set()
        urls: list[str] = []
        for url in md_urls + bare_urls:
            cleaned = url.rstrip(".,;:!?\"')>}]")
            if cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)
        return urls

    @classmethod
    def _find_hallucinated_urls(cls, text: str, source_text: str) -> list[str]:
        """Return URLs in text that don't appear verbatim in the source text."""
        urls = cls._extract_urls(text)
        if not urls:
            return []
        return [url for url in urls if url not in source_text]

    def _get_source_text(self) -> str:
        """Combined tool result text from the current run for URL validation."""
        return "\n".join(self._tool_result_text)

    # ── Message building ─────────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Build message list for Ollama chat API.

        The system_prompt is the full prompt body (identity, context,
        instructions) built by each agent's _build_system_prompt method.
        This method only prepends the timestamp.
        """
        effective = system_prompt or self.system_prompt
        now = datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M %p UTC")
        system_content = f"Current date and time: {now}\n\n{effective}"

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_content).to_dict()]

        if history:
            for role, content in history:
                messages.append(ChatMessage(role=MessageRole(role), content=content).to_dict())

        messages.append(ChatMessage(role=MessageRole.USER, content=prompt).to_dict())
        return messages

    def _build_tool_summary(self) -> str:
        """Build a dynamic tool summary from registered tools for prompt injection."""
        tools = self._tool_registry.get_all()
        if not tools:
            return ""
        names = [t.name for t in tools]
        logger.debug("Injecting tool summary into prompt: %s", ", ".join(names))
        lines = [f"- **{t.name}**: {t.description}" for t in tools]
        return "\n".join(lines)

    # ── System prompt building (template method pattern) ─────────────────

    async def _build_system_prompt(self, user: str) -> str:
        """Build the full system prompt body. Override per agent.

        Each agent composes its prompt from building blocks below.
        The timestamp is prepended by _build_messages — don't include it here.
        """
        sections = [
            self._identity_section(),
            self._context_block(
                self._profile_section(user),
                self._history_section(user),
                self._thought_section(user),
                self._dislike_section(user),
            ),
            self._instructions_section(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ── Building blocks ───────────────────────────────────────────────────

    def _identity_section(self) -> str:
        """## Identity — Penny's voice and personality."""
        return f"## Identity\n{Prompt.PENNY_IDENTITY}"

    def _instructions_section(self, override: str | None = None) -> str:
        """## Instructions — agent-specific prompt with tool descriptions."""
        prompt = override or self.system_prompt
        format_args: dict = {}
        if "{tools}" in prompt:
            format_args["tools"] = self._build_tool_summary()
        if "{max_tool_calls}" in prompt:
            assert self._multi_tool is not None, "{max_tool_calls} in prompt but no multi_tool"
            format_args["max_tool_calls"] = self._multi_tool._max_calls
        if format_args:
            prompt = prompt.format(**format_args)
        return f"## Instructions\n{prompt}"

    @staticmethod
    def _context_block(*sections: str | None) -> str | None:
        """Wrap non-None sections under a ## Context header."""
        parts = [s for s in sections if s]
        if not parts:
            return None
        return "## Context\n" + "\n\n".join(parts)

    def _profile_section(self, sender: str) -> str | None:
        """### User Profile — user name."""
        try:
            user_info = self.db.users.get_info(sender)
            if not user_info:
                return None

            logger.debug("Built profile context for %s", sender)
            return f"### User Profile\nThe user's name is {user_info.name}."
        except Exception:
            return None

    def _history_section(self, sender: str) -> str | None:
        """Build conversation history with weekly summaries and daily details.

        Weekly rollups replace their constituent daily entries to avoid
        duplication. Only daily entries outside any weekly range are shown.

        Format:
            Week of Mar 3:
            - weekly theme 1
            Mar 15:
            - daily topic 1
            Today:
            - daily topic 2
        """
        try:
            lines: list[str] = []
            weekly_entries = self._get_weekly_entries(sender)
            lines.extend(self._format_weekly_entries(weekly_entries))
            lines.extend(self._format_daily_entries(sender, weekly_entries))
            if not lines:
                return None

            logger.debug("Built history context")
            return "### Conversation History\n" + "\n".join(lines)
        except Exception:
            logger.warning("History context retrieval failed, proceeding without")
            return None

    def _get_weekly_entries(self, sender: str) -> list:
        """Fetch weekly history entries."""
        weekly_limit = int(self.config.runtime.WEEKLY_CONTEXT_LIMIT)
        return self.db.history.get_recent(
            sender, PennyConstants.HistoryDuration.WEEKLY, limit=weekly_limit
        )

    def _format_weekly_entries(self, entries: list) -> list[str]:
        """Format weekly history entries with 'Week of' date labels."""
        lines: list[str] = []
        for entry in entries:
            date_label = f"Week of {entry.period_start.strftime('%b %-d')}"
            topics = self._extract_topic_lines(entry.topics)
            if topics:
                lines.append(f"{date_label}:")
                lines.extend(f"- {t}" for t in topics)
        return lines

    def _format_daily_entries(self, sender: str, weekly_entries: list) -> list[str]:
        """Format daily history entries, skipping days covered by a weekly rollup."""
        daily_limit = int(self.config.runtime.HISTORY_CONTEXT_LIMIT)
        entries = self.db.history.get_recent(
            sender, PennyConstants.HistoryDuration.DAILY, limit=daily_limit
        )
        today = self._midnight_today()
        weekly_ranges = [(w.period_start, w.period_end) for w in weekly_entries]
        lines: list[str] = []
        for entry in entries:
            if self._covered_by_weekly(entry.period_start, weekly_ranges):
                continue
            is_today = entry.period_start == today
            date_label = "Today" if is_today else entry.period_start.strftime("%b %-d")
            topics = self._extract_topic_lines(entry.topics)
            if topics:
                lines.append(f"{date_label}:")
                lines.extend(f"- {t}" for t in topics)
        return lines

    @staticmethod
    def _covered_by_weekly(day_start, weekly_ranges: list[tuple]) -> bool:
        """Check if a daily entry falls within a completed weekly rollup."""
        return any(week_start <= day_start < week_end for week_start, week_end in weekly_ranges)

    @staticmethod
    def _extract_topic_lines(topics: str) -> list[str]:
        """Parse topic bullet text into clean topic strings."""
        result: list[str] = []
        for line in topics.strip().splitlines():
            topic = line.strip().lstrip("- ").strip()
            if topic:
                result.append(topic)
        return result

    def _thought_section(self, sender: str) -> str | None:
        """Build recent thinking summary context. Overridden by ChatAgent and ThinkingAgent."""
        try:
            thoughts = self.db.thoughts.get_recent(sender, limit=self.THOUGHT_CONTEXT_LIMIT)
            if not thoughts:
                return None
            lines = [t.content for t in thoughts]
            logger.debug("Built thought context (%d thoughts)", len(thoughts))
            return "### Recent Background Thinking\n" + "\n\n---\n\n".join(lines)
        except Exception:
            logger.warning("Thought context retrieval failed, proceeding without")
            return None

    def _dislike_section(self, user: str) -> str | None:
        """Build textual list of topics the user dislikes."""
        try:
            prefs = self.db.preferences.get_for_user(user)
            negative = [
                p.content for p in prefs if p.valence == PennyConstants.PreferenceValence.NEGATIVE
            ]
            if not negative:
                return None
            seen: set[str] = set()
            unique: list[str] = []
            for text in negative:
                key = text.strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(text.strip())
            lines = "\n".join(f"- {t}" for t in unique)
            return f"### Topics to Avoid\n{lines}"
        except Exception:
            logger.warning("Dislike context retrieval failed, proceeding without")
            return None

    def _build_conversation(self, sender: str) -> list[tuple[str, str]]:
        """Build conversation history as strict user/assistant alternation.

        Only includes messages since the last history rollup (or midnight
        if no rollup exists), so rolled-up content isn't duplicated.
        Consecutive same-role messages are merged with newlines to maintain
        valid turn structure for the model.
        """
        conversation: list[tuple[str, str]] = []
        try:
            limit = int(self.config.runtime.MESSAGE_CONTEXT_LIMIT)
            since = self._conversation_start(sender)
            messages = self.db.messages.get_messages_since(sender, since=since, limit=limit)
            for msg in messages:
                role = (
                    MessageRole.USER
                    if msg.direction == PennyConstants.MessageDirection.INCOMING
                    else MessageRole.ASSISTANT
                )
                if conversation and conversation[-1][0] == role:
                    prev_role, prev_content = conversation[-1]
                    conversation[-1] = (prev_role, prev_content + "\n" + msg.content)
                else:
                    conversation.append((role, msg.content))
            if conversation:
                logger.debug("Built conversation (%d turns since %s)", len(conversation), since)
        except Exception:
            logger.warning("Conversation building failed, proceeding without")
        return conversation

    def _conversation_start(self, sender: str) -> datetime:
        """Determine where raw message history should begin.

        Returns the latest rollup's period_end (so we don't duplicate
        summarized content), or midnight today if no rollup exists.
        """
        latest = self.db.history.get_latest(sender, PennyConstants.HistoryDuration.DAILY)
        if latest is not None:
            return latest.period_end
        return self._midnight_today()

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _midnight_today() -> datetime:
        """Return midnight UTC for today as a naive datetime.

        Naive because SQLite strips timezone info — all stored datetimes are naive UTC.
        """
        return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Remove this agent from the instance registry."""
        if self in Agent._instances:
            Agent._instances.remove(self)

    @classmethod
    async def close_all(cls) -> None:
        """Close all agent instances."""
        for agent in cls._instances[:]:
            await agent.close()
