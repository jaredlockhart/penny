"""Base Agent class with agentic loop and context building."""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse as _urlparse
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from similarity.embeddings import cosine_similarity

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole, ToolCallRecord
from penny.config import Config
from penny.constants import PennyConstants, ValidationReason
from penny.database import Database
from penny.llm import LlmClient
from penny.llm.embeddings import deserialize_embedding
from penny.llm.models import LlmError
from penny.llm.similarity import centrality_scores
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool, ToolCall, ToolExecutor, ToolRegistry
from penny.tools.browse import BrowseTool
from penny.tools.models import SearchResult

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


def _build_strong_nudge(messages: list[dict]) -> str:
    """Build a context-aware nudge that includes the original user question.

    Called when many preceding tool calls may have saturated the model's context.
    Uses forceful language to break the model out of search-fixation loops.
    Including the original question gives the model a clear target after heavy tool use.
    """
    user_messages = [
        m["content"]
        for m in messages
        if m.get("role") == MessageRole.USER and not m["content"].startswith("STOP")
    ]
    original_question = user_messages[-1]
    return Prompt.FINAL_STEP_NUDGE.format(original_question=original_question)


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

    Agents receive shared LlmClient instances — foreground (fast, user-facing)
    and background (smart, processing). Callers create and own the clients;
    agents just hold references.
    """

    _instances: list[Agent] = []

    THOUGHT_CONTEXT_LIMIT = PennyConstants.THOUGHT_CONTEXT_LIMIT
    PREFERRED_POOL_SIZE = PennyConstants.PREFERRED_POOL_SIZE
    name: str = "Agent"

    def __init__(
        self,
        system_prompt: str,
        model_client: LlmClient,
        tools: list[Tool],
        db: Database,
        config: Config,
        tool_timeout: float = 60.0,
        vision_model_client: LlmClient | None = None,
        embedding_model_client: LlmClient | None = None,
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
        self._browse_tool: BrowseTool | None = None
        self._browse_provider: Callable[[], Any] | None = None
        self._current_user: str | None = None
        self._tool_result_text: list[str] = []
        self._tool_result_images: list[str] = []

        self._tool_registry = ToolRegistry()
        for tool in self.tools:
            self._tool_registry.register(tool)

        self._tool_executor = ToolExecutor(self._tool_registry, timeout=tool_timeout)
        self._keep_tools_on_final_step = False
        # Lazy in-memory centrality cache for related-message retrieval, keyed by
        # sender. Computed once on first retrieval per process and reused. Drifts
        # as new messages arrive — acceptable for the MVP; revisit (DB column,
        # background refresh) if precision degrades or the corpus grows past a
        # few thousand messages where O(N²) recomputation becomes painful.
        self._message_centrality_cache: dict[str, dict[int, float]] = {}
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
            run_id = uuid.uuid4().hex
            prompt_type = self.get_prompt_type()
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
                    run_id=run_id,
                    prompt_type=prompt_type,
                )
            finally:
                if tool_cleanup:
                    await tool_cleanup()
            did_work = await self.after_run(user, run_id, prompt_type)
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

    def get_prompt_type(self) -> str | None:
        """Return the prompt type for the current cycle. Override in subclasses."""
        return None

    async def after_run(self, user: str, run_id: str, prompt_type: str | None = None) -> bool:
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
        run_id: str | None = None,
        prompt_type: str | None = None,
    ) -> ControllerResponse:
        """Run the agentic loop — prompt in, response out."""
        if run_id is None:
            run_id = uuid.uuid4().hex
        self._tool_result_text = []
        self._tool_result_images = []
        messages = self._build_messages(prompt, history, system_prompt)
        tools = self._tool_registry.get_ollama_tools()
        return await self._run_agentic_loop(
            messages, tools, max_steps, on_tool_start, run_id, prompt_type
        )

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
        run_id: str | None = None,
        prompt_type: str | None = None,
    ) -> ControllerResponse:
        """Execute the step loop: call model, process tool calls, or return final answer."""
        attachments: list[str] = []
        source_urls: list[str] = []
        called_tools: set[tuple[str, ...]] = set()
        tool_call_records: list[ToolCallRecord] = []

        for step in range(steps):
            logger.info("Agent step %d/%d", step + 1, steps)

            is_final_step = step == steps - 1
            strip_tools = is_final_step and not self._keep_tools_on_final_step
            step_tools = [] if strip_tools else tools
            if strip_tools:
                logger.debug("Final step — tools removed, model must produce text")

            response = await self._call_model_validated(messages, step_tools, run_id, prompt_type)
            if response is None:
                return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)

            if response.has_tool_calls:
                result = await self._process_tool_calls(response, called_tools, on_tool_start)
                messages.extend(result.messages)
                tool_call_records.extend(result.records)
                source_urls.extend(result.source_urls)
                attachments.extend(result.attachments)
                self._tool_result_images.extend(result.attachments)
                await self.after_step(result.records, result.messages, messages)
                if self.should_stop_loop(result.records):
                    logger.info("Loop stop requested after step %d/%d", step + 1, steps)
                    break
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
                continue

            if await self.handle_text_step(response, messages, step, is_final_step):
                continue

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

    async def after_step(
        self,
        step_records: list[ToolCallRecord],
        step_messages: list[dict],
        conversation: list[dict] | None = None,
    ) -> None:
        """Capture tool result text for URL validation. Override in subclasses (call super)."""
        for message in step_messages:
            if message.get("role") == MessageRole.TOOL:
                content = message.get("content", "")
                if content:
                    self._tool_result_text.append(content)

    def should_stop_loop(self, step_records: list[ToolCallRecord]) -> bool:
        """Check if the loop should stop early. Override in subclasses."""
        return False

    async def _call_model_validated(
        self,
        messages: list[dict],
        tools: list[dict],
        run_id: str | None = None,
        prompt_type: str | None = None,
    ):
        """Call the model, retrying on invalid outputs.

        Checks for (in order): XML markup, empty content, refusal, hallucinated URLs.
        Each invalid output type gets one retry. Tool call responses are returned
        immediately without validation. When tools are stripped (None) but the model
        hallucinates tool calls, they are cleared and content falls through to
        normal validation — which triggers the appropriate nudge for empty responses.
        """
        max_retries = PennyConstants.RESPONSE_VALIDATION_RETRIES
        effective_tools = tools if tools else None
        retried: set[ValidationReason] = set()

        for attempt in range(max_retries):
            try:
                response = await self._model_client.chat(
                    messages=messages,
                    tools=effective_tools,
                    agent_name=self.name,
                    prompt_type=prompt_type,
                    run_id=run_id,
                )
            except Exception as exception:
                logger.error("Error calling Ollama: %s", exception)
                return None

            # Tool calls with tools available — return immediately, no validation
            if response.has_tool_calls and effective_tools is not None:
                return response

            # Hallucinated tool calls (tools stripped) — clear and validate content
            if response.has_tool_calls and effective_tools is None:
                logger.warning("Model hallucinated tool calls without tools — stripping")
                response.message.tool_calls = None

            self.on_response(response)
            content = response.content.strip()
            reason = self._check_response(content, retried)
            if reason is None:
                return response

            retried.add(reason)
            logger.warning(
                "Invalid response (%s) on attempt %d/%d",
                reason,
                attempt + 1,
                max_retries,
            )

            # Append the bad response so the model sees what it produced
            messages.append(response.message.to_input_message())

            # Empty content: nudge depends on whether the model still has tools
            if reason == ValidationReason.EMPTY:
                if effective_tools is None:
                    # Final step, tools stripped — force synthesis
                    nudge = _build_strong_nudge(messages)
                else:
                    # Mid-loop, tools still available — gentle nudge to continue
                    nudge = Prompt.CONTINUE_NUDGE
                messages.append({"role": MessageRole.USER, "content": nudge})

        return response

    def _check_response(
        self, content: str, already_retried: set[ValidationReason]
    ) -> ValidationReason | None:
        """Check a text response for problems. Returns reason or None if valid."""
        if _has_xml_tags(content) and ValidationReason.XML not in already_retried:
            return ValidationReason.XML

        effective_content, _ = _strip_think_tags(content)
        letter_count = sum(1 for c in effective_content if c.isalpha())
        if (
            letter_count < PennyConstants.MIN_RESPONSE_LETTERS
            and ValidationReason.EMPTY not in already_retried
        ):
            return ValidationReason.EMPTY

        if (
            effective_content
            and self._is_refusal(effective_content)
            and ValidationReason.REFUSAL not in already_retried
        ):
            return ValidationReason.REFUSAL

        source_text = self._get_source_text()
        if source_text and effective_content:
            bad_urls = self._find_hallucinated_urls(effective_content, source_text)
            if bad_urls and ValidationReason.HALLUCINATED_URLS not in already_retried:
                logger.warning(
                    "Hallucinated URL(s): %s",
                    ", ".join(url[:80] for url in bad_urls),
                )
                return ValidationReason.HALLUCINATED_URLS

        return None

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

    def get_tools(self, user: str) -> list[Tool]:
        """Build tool list for this agent.

        When max_queries_key is set, builds a fresh BrowseTool each cycle
        so runtime config changes take effect immediately.
        """
        if self._max_queries_key is not None:
            return [self._build_browse_tool()]
        return []

    def _build_browse_tool(self) -> BrowseTool:
        """Build a fresh BrowseTool from config, updating self._browse_tool."""
        assert self._max_queries_key is not None
        max_calls = int(getattr(self.config.runtime, self._max_queries_key))
        search_url = str(self.config.runtime.SEARCH_URL)
        tool = BrowseTool(max_calls=max_calls, search_url=search_url)
        if self._browse_provider:
            tool.set_browse_provider(self._browse_provider)
        self._browse_tool = tool
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
        pending: list[tuple[str, str, dict, str | None]] = []
        for tool_call in response.message.tool_calls or []:
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            # Pop reasoning before dedup (same args + different reasoning = repeat)
            reasoning = arguments.pop("reasoning", None)
            call_key = self._make_call_key(tool_name, arguments)

            if not self.allow_repeat_tools and call_key in called_tools:
                logger.info("Skipping repeat: %s(%s)", tool_name, arguments)
                repeat_msg = "You already made this exact tool call. Try a different query or tool."
                messages.append(
                    {"role": MessageRole.TOOL, "content": repeat_msg, "tool_call_id": tool_call_id}
                )
                continue

            called_tools.add(call_key)
            pending.append((tool_call_id, tool_name, arguments, reasoning))

        # Fire on_tool_start once with all pending tools so the UI can show
        # a combined status (e.g. "Searching A + Searching B") for parallel calls.
        if on_tool_start and pending:
            try:
                await on_tool_start([(name, dict(args)) for _, name, args, _ in pending])
            except Exception:
                logger.debug("on_tool_start callback failed")

        # Execute all valid tool calls in parallel.
        results = await asyncio.gather(
            *[
                self._execute_single_tool(name, args, reasoning)
                for _, name, args, reasoning in pending
            ]
        )

        for (tool_call_id, _, _, _), (result_str, record, urls, image) in zip(
            pending, results, strict=True
        ):
            records.append(record)
            source_urls.extend(urls)
            if image:
                attachments.append(image)
            messages.append(
                {"role": MessageRole.TOOL, "content": result_str, "tool_call_id": tool_call_id}
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
            record.failed = _is_tool_result_failed(result_str)
            logger.debug("Tool result: %s", result_str[:200])
            return result_str, record, urls, image
        result_str = str(tool_result.result)
        record.failed = _is_tool_result_failed(result_str)
        logger.debug("Tool result: %s", result_str[:200])
        return result_str, record, [], None

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

    def _thought_section(self, sender: str) -> str | None:
        """Build recent thinking summary context. Overridden by ChatAgent and ThinkingAgent."""
        try:
            thoughts = self.db.thoughts.get_recent(sender, limit=self.THOUGHT_CONTEXT_LIMIT)
            if not thoughts:
                return None
            lines = [t.content for t in thoughts]
            logger.debug("Built thought context (%d thoughts)", len(thoughts))
            return "### Recent Background Thinking\n" + PennyConstants.SECTION_SEPARATOR.join(lines)
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

    async def _related_knowledge_section(
        self, conversation_embeddings: list[list[float]] | None
    ) -> str | None:
        """Retrieve knowledge entries relevant to the current conversation."""
        if not conversation_embeddings:
            return None
        try:
            return self._build_related_knowledge(conversation_embeddings)
        except Exception:
            logger.warning("Knowledge retrieval failed, proceeding without")
            return None

    def _build_related_knowledge(self, conversation_embeddings: list[list[float]]) -> str | None:
        """Score knowledge entries against conversation, return top N formatted.

        Each candidate is scored as `max(weighted_decay, current_message_cosine)` so
        that strong direct matches stand on their own merit while vague follow-ups
        still benefit from conversation-context drift. A floor suppresses noise on
        greetings and topics the corpus doesn't cover.
        """
        entries = self.db.knowledge.get_with_embeddings()
        if not entries:
            return None

        scored = self._score_knowledge_hybrid(conversation_embeddings, entries)
        top = self._filter_knowledge_by_floor(scored)
        if not top:
            return None
        return self._format_knowledge(top)

    def _score_knowledge_hybrid(
        self, conversation_embeddings: list[list[float]], entries: list
    ) -> list[tuple[float, Any]]:
        """Score each entry as max(weighted_decay, cosine_to_current), sorted desc."""
        current = conversation_embeddings[-1]
        scored: list[tuple[float, Any]] = []
        for entry in entries:
            candidate_vec = deserialize_embedding(entry.embedding)
            weighted = self._weighted_similarity(conversation_embeddings, candidate_vec)
            current_cosine = cosine_similarity(current, candidate_vec)
            scored.append((max(weighted, current_cosine), entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def _filter_knowledge_by_floor(self, scored: list[tuple[float, Any]]) -> list:
        """Apply absolute score floor and entry limit."""
        floor = float(self.config.runtime.RELATED_KNOWLEDGE_SCORE_FLOOR)
        limit = int(self.config.runtime.RELATED_KNOWLEDGE_LIMIT)
        return [entry for score, entry in scored[:limit] if score >= floor]

    async def _related_messages_section(
        self, sender: str, conversation_embeddings: list[list[float]] | None
    ) -> str | None:
        """Retrieve past user messages semantically similar to the conversation."""
        if not conversation_embeddings:
            return None
        try:
            return self._build_related_messages(sender, conversation_embeddings)
        except Exception:
            logger.warning("Related messages retrieval failed, proceeding without")
            return None

    def _build_related_messages(
        self, sender: str, conversation_embeddings: list[list[float]]
    ) -> str | None:
        """Score past messages by current-message cosine minus a centrality penalty.

        Unlike knowledge retrieval (which uses weighted decay over the conversation
        window to capture topic drift), message retrieval scores against ONLY the
        current user message — we want messages similar to what's being asked right
        now, not adjacent topics from earlier in the thread.

        Each candidate's score is `cosine_to_current - α * centrality`, where
        centrality is the message's mean cosine to the rest of the corpus (a
        centroid-magnet penalty that suppresses generic boilerplate like
        "hi penny" / "Hey Penny what are some recent..." which would otherwise
        leak into every unrelated query).

        Selection is adaptive: a cluster-strength gate suppresses noise plateaus
        entirely (novel topics with no real history), and the cutoff combines a
        relative band against the cluster center with an absolute floor — strong
        clusters return many messages, weak clusters return few, no cluster
        returns nothing.
        """
        messages = self.db.messages.get_incoming_with_embeddings(sender)
        if not messages:
            return None

        candidates = self._eligible_message_candidates(sender, messages)
        if not candidates:
            return None

        centralities = self._get_message_centralities(sender, messages)
        scored = self._score_messages_with_centrality_penalty(
            conversation_embeddings[-1], candidates, centralities
        )
        cutoff = self._related_messages_cutoff(scored)
        if cutoff is None:
            return None

        limit = int(self.config.runtime.RELATED_MESSAGES_LIMIT)
        top = [message for score, message in scored if score >= cutoff][:limit]
        if not top:
            return None
        expanded = self._expand_with_neighbors(sender, top)
        return self._format_related_messages(expanded)

    def _expand_with_neighbors(self, sender: str, hits: list) -> list:
        """Pull user messages within ±N minutes of each hit, deduped and excluding
        the current conversation window.

        Captures conversational follow-ups that share no entity overlap with the
        current message ("yeah exactly i can't wait to try it") but live in the
        same conversation as a real hit. Single pass — neighbors are not themselves
        expanded.
        """
        window = timedelta(minutes=PennyConstants.RELATED_MESSAGES_NEIGHBOR_WINDOW_MINUTES)
        conversation_ids = self._get_conversation_message_ids(sender)
        seen_ids: set[int] = {hit.id for hit in hits if hit.id is not None}
        seen_content: set[str] = {hit.content.strip().lower() for hit in hits}
        expanded = list(hits)
        for hit in hits:
            for neighbor in self.db.messages.get_user_messages_in_window(
                sender, hit.timestamp, window
            ):
                if neighbor.id is None or neighbor.id in seen_ids:
                    continue
                if neighbor.id in conversation_ids:
                    continue
                content_key = neighbor.content.strip().lower()
                if content_key in seen_content:
                    continue
                seen_ids.add(neighbor.id)
                seen_content.add(content_key)
                expanded.append(neighbor)
        return expanded

    def _eligible_message_candidates(self, sender: str, messages: list) -> list:
        """Filter to messages outside the current conversation, deduped by content."""
        conversation_ids = self._get_conversation_message_ids(sender)
        seen: set[str] = set()
        candidates: list = []
        for message in messages:
            if not message.embedding or message.id is None:
                continue
            if message.id in conversation_ids:
                continue
            key = message.content.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(message)
        return candidates

    def _get_message_centralities(self, sender: str, messages: list) -> dict[int, float]:
        """Lazily compute and cache per-sender message centralities.

        Centrality = mean cosine to every other message in the sender's corpus.
        Computed once on first retrieval and cached for the process lifetime.
        Drifts as new messages arrive — acceptable trade-off for the MVP.
        """
        cached = self._message_centrality_cache.get(sender)
        if cached is not None:
            return cached
        vecs: dict[int, list[float]] = {
            message.id: deserialize_embedding(message.embedding)
            for message in messages
            if message.id is not None and message.embedding
        }
        cache = centrality_scores(vecs)
        self._message_centrality_cache[sender] = cache
        return cache

    @staticmethod
    def _score_messages_with_centrality_penalty(
        current_embedding: list[float],
        candidates: list,
        centralities: dict[int, float],
    ) -> list[tuple[float, Any]]:
        """Score each candidate as `cosine - α * centrality`, sorted descending."""
        scored: list[tuple[float, Any]] = []
        for message in candidates:
            cosine = cosine_similarity(current_embedding, deserialize_embedding(message.embedding))
            penalty = PennyConstants.RELATED_MESSAGES_CENTRALITY_PENALTY * centralities.get(
                message.id, 0.0
            )
            scored.append((cosine - penalty, message))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    @staticmethod
    def _related_messages_cutoff(scored: list[tuple[float, Any]]) -> float | None:
        """Compute adaptive cutoff for related-message selection.

        With enough candidates (≥20), apply a cluster-strength gate that suppresses
        flat noise plateaus entirely, then combine a relative band against the
        cluster center with an absolute floor. With fewer candidates (cold start),
        skip the gate and use just the absolute floor.

        Returns None if the noise plateau gate fires (no real cluster present).
        """
        if not scored:
            return None
        head_size = PennyConstants.RELATED_MESSAGES_GATE_HEAD_SIZE
        sample_size = PennyConstants.RELATED_MESSAGES_GATE_SAMPLE_SIZE
        if len(scored) >= sample_size:
            head_mean = sum(score for score, _ in scored[:head_size]) / head_size
            sample_mean = sum(score for score, _ in scored[:sample_size]) / sample_size
            if (
                sample_mean <= 0
                or head_mean / sample_mean < PennyConstants.RELATED_MESSAGES_CLUSTER_GATE
            ):
                return None
            return max(
                head_mean * PennyConstants.RELATED_MESSAGES_RELATIVE_RATIO,
                PennyConstants.RELATED_MESSAGES_ABSOLUTE_FLOOR,
            )
        return PennyConstants.RELATED_MESSAGES_ABSOLUTE_FLOOR

    # ── Weighted scoring helpers ──────────────────────────────────────────

    async def _embed_conversation(self, sender: str, content: str) -> list[list[float]] | None:
        """Embed each message in the current conversation + the new message."""
        if not self._embedding_model_client:
            return None
        conversation = self._build_conversation(sender)
        texts = [text for _role, text in conversation] + [content]
        try:
            return await self._embedding_model_client.embed(texts)
        except LlmError:
            logger.warning("Conversation embedding failed")
            return None

    @staticmethod
    def _weighted_similarity(
        message_embeddings: list[list[float]],
        candidate_embedding: list[float],
        decay: float = 0.5,
    ) -> float:
        """Compute exponentially-decayed weighted average of cosine similarities.

        Most recent message gets weight 1.0, previous gets decay,
        before that decay^2, etc.
        """
        count = len(message_embeddings)
        if count == 0:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for index, message_vec in enumerate(message_embeddings):
            age = count - 1 - index
            weight = decay**age
            total += cosine_similarity(message_vec, candidate_embedding) * weight
            weight_sum += weight
        return total / weight_sum

    # ── Context formatters ────────────────────────────────────────────────

    @staticmethod
    def _format_knowledge(entries: list) -> str:
        """Format knowledge entries as titled paragraphs."""
        sections: list[str] = []
        for entry in entries:
            sections.append(f"{entry.title}\n{entry.url}\n{entry.summary}")
        if not sections:
            return "### Knowledge"
        return "### Knowledge\n" + "\n\n".join(sections)

    @staticmethod
    def _format_related_messages(messages: list) -> str:
        """Format messages as dated quotes, ordered chronologically."""
        messages_sorted = sorted(messages, key=lambda message: message.timestamp)
        lines: list[str] = []
        for message in messages_sorted:
            date_label = message.timestamp.strftime("%b %-d")
            lines.append(f'{date_label}: "{message.content}"')
        if not lines:
            return "### Related Past Messages"
        return "### Related Past Messages\n" + "\n".join(lines)

    def _get_conversation_message_ids(self, sender: str) -> set[int]:
        """Get IDs of messages in the current conversation window (last N)."""
        try:
            limit = int(self.config.runtime.MESSAGE_CONTEXT_LIMIT)
            messages = self.db.messages.get_messages_since(sender, since=datetime.min, limit=limit)
            return {msg.id for msg in messages if msg.id is not None}
        except Exception:
            return set()

    def _build_conversation(self, sender: str) -> list[tuple[str, str]]:
        """Build conversation history as strict user/assistant alternation.

        Fetches the last N messages (no time boundary). Consecutive same-role
        messages are merged with newlines to maintain valid turn structure.
        """
        conversation: list[tuple[str, str]] = []
        try:
            limit = int(self.config.runtime.MESSAGE_CONTEXT_LIMIT)
            messages = self.db.messages.get_messages_since(sender, since=datetime.min, limit=limit)
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
                logger.debug("Built conversation (%d turns)", len(conversation))
        except Exception:
            logger.warning("Conversation building failed, proceeding without")
        return conversation

    # ── Utilities ────────────────────────────────────────────────────────

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
