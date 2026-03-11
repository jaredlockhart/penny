"""Base Agent class with agentic loop and context building."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from penny.agents.models import ChatMessage, ControllerResponse, MessageRole, ToolCallRecord
from penny.config import Config
from penny.constants import PennyConstants
from penny.database import Database
from penny.ollama import OllamaClient
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import SearchTool, Tool, ToolCall, ToolExecutor, ToolRegistry
from penny.tools.models import SearchResult

logger = logging.getLogger(__name__)


# Matches paired XML-like tags in content, e.g. <function=search>...</function>
# or <tools><search>...</search></tools>
_XML_TAG_PATTERN = re.compile(r"<[a-zA-Z]\w*[\s=>].*</[a-zA-Z]\w*>", re.DOTALL)


def _has_xml_tags(content: str) -> bool:
    """Return True if content contains XML-like tag pairs."""
    return bool(_XML_TAG_PATTERN.search(content))


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

    THOUGHT_CONTEXT_LIMIT = 10
    PREFERRED_POOL_SIZE = 5
    name: str = "Agent"

    def __init__(
        self,
        system_prompt: str,
        model_client: OllamaClient,
        tools: list[Tool],
        db: Database,
        config: Config,
        max_steps: int = 5,
        tool_timeout: float = 60.0,
        vision_model_client: OllamaClient | None = None,
        embedding_model_client: OllamaClient | None = None,
        allow_repeat_tools: bool = False,
        search_tool: Tool | None = None,
        news_tool: Tool | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.tools = tools
        self.db = db
        self.max_steps = max_steps
        self.allow_repeat_tools = allow_repeat_tools

        self._model_client = model_client
        self._vision_model_client = vision_model_client
        self._embedding_model_client = embedding_model_client

        self._search_tool = search_tool
        self._news_tool = news_tool
        self._current_user: str | None = None

        self._tool_registry = ToolRegistry()
        for tool in self.tools:
            self._tool_registry.register(tool)

        self._tool_executor = ToolExecutor(self._tool_registry, timeout=tool_timeout)

        Agent._instances.append(self)

        logger.info(
            "Initialized agent: model=%s, tools=%d, max_steps=%d",
            self._model_client.model,
            len(self.tools),
            max_steps,
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
            context = await self.get_context(user)
            history = self.get_history(user)
            await self.run(prompt=prompt, context=context, history=history)
            did_work = await self.after_run(user)
            logger.info("%s complete for %s", self.name, user)
            return did_work
        except Exception:
            logger.exception("%s failed for %s", self.name, user)
            return False
        finally:
            self._current_user = None

    # ── Override hooks ───────────────────────────────────────────────────

    def get_users(self) -> list[str]:
        """Return users to process. Override to filter."""
        return self.db.users.get_all_senders()

    async def get_prompt(self, user: str) -> str | None:
        """Build the prompt for the agentic loop. Return None to skip this user."""
        return None

    async def get_context(self, user: str) -> str:
        """Build context text for the system prompt. Override for custom context."""
        context, _ = await self._build_context(user)
        return context

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
        history: list[tuple[str, str]] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        context: str | None = None,
    ) -> ControllerResponse:
        """Run the agentic loop — prompt in, response out."""
        messages = self._build_messages(prompt, history, system_prompt, context=context)
        tools = self._tool_registry.get_ollama_tools()
        steps = max_steps if max_steps is not None else self.max_steps
        return await self._run_agentic_loop(messages, tools, steps)

    # ── Agentic loop internals ───────────────────────────────────────────

    async def _run_agentic_loop(
        self,
        messages: list[dict],
        tools: list[dict],
        steps: int,
    ) -> ControllerResponse:
        """Execute the step loop: call model, process tool calls, or return final answer."""
        attachments: list[str] = []
        source_urls: list[str] = []
        called_tools: set[tuple[str, ...]] = set()
        tool_call_records: list[ToolCallRecord] = []
        empty_retries: int = 0

        for step in range(steps):
            logger.info("Agent step %d/%d", step + 1, steps)

            is_final_step = step == steps - 1
            step_tools = [] if is_final_step else tools
            if is_final_step:
                logger.debug("Final step — tools removed, model must produce text")

            response = await self._call_model_with_xml_retry(messages, step_tools)
            if response is None:
                return ControllerResponse(answer=PennyResponse.AGENT_MODEL_ERROR)

            self.on_response(response)

            if response.has_tool_calls:
                result = await self._process_tool_calls(response, called_tools)
                messages.extend(result.messages)
                tool_call_records.extend(result.records)
                source_urls.extend(result.source_urls)
                attachments.extend(result.attachments)
                await self.after_step(result.records, messages)
                if self.should_stop_loop(result.records):
                    logger.info("Loop stop requested after step %d/%d", step + 1, steps)
                    break
                continue

            if await self.handle_text_step(response, messages, step, is_final_step):
                continue

            if not response.content.strip() and empty_retries == 0:
                empty_retries += 1
                logger.warning(
                    "Model returned empty content on step %d/%d; requesting text output",
                    step + 1,
                    steps,
                )
                messages.append(response.message.to_input_message())
                messages.append(
                    {"role": MessageRole.USER, "content": "Please provide your response."}
                )
                if not is_final_step:
                    continue
                # On the final step, retry directly — can't extend a for-range loop
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
        """Hook called after each step's tool calls. Override in subclasses."""

    def should_stop_loop(self, step_records: list[ToolCallRecord]) -> bool:
        """Check if the loop should stop early. Override in subclasses."""
        return False

    async def _call_model_with_xml_retry(self, messages: list[dict], tools: list[dict]):
        """Call the model, retrying if it emits XML markup instead of structured tool calls."""
        max_xml_retries = 3
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

    # ── Tool management ──────────────────────────────────────────────────

    def get_tools(self, user: str) -> list[Tool]:
        """Build tool list for this agent. Override in subclasses for custom tools."""
        tools: list[Tool] = []
        if self._search_tool:
            tools.append(self._search_tool)
        if self._news_tool:
            tools.append(self._news_tool)
        return tools

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
    ) -> _StepResult:
        """Process all tool calls from a model response. Returns results to append."""
        logger.info("Model requested %d tool call(s)", len(response.message.tool_calls or []))
        messages: list[dict] = [response.message.to_input_message()]
        records: list[ToolCallRecord] = []
        source_urls: list[str] = []
        attachments: list[str] = []

        for ollama_tool_call in response.message.tool_calls or []:
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
            result_str, record, urls, image = await self._execute_single_tool(
                tool_name, arguments, reasoning
            )
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
            logger.debug("Tool result: %s", result_str[:200])
            return result_str, record, [], None

        if isinstance(tool_result.result, SearchResult):
            result_str, urls, image = self._format_search_result(tool_result.result)
            logger.debug("Tool result: %s", result_str[:200])
            return result_str, record, urls, image

        result_str = str(tool_result.result)
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

    # ── Message building ─────────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        context: str | None = None,
    ) -> list[dict]:
        """Build message list for Ollama chat API.

        Args:
            prompt: The user message/prompt to respond to
            history: Optional conversation history as (role, content) tuples
            system_prompt: Optional system prompt override
            context: Optional context text appended to system prompt (profile, events, etc.)

        Returns:
            List of message dicts for Ollama chat API
        """
        messages = []

        effective_prompt = system_prompt or self.system_prompt
        now = datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M %p UTC")

        # Build system prompt: timestamp → identity → context → agent-specific prompt
        system_parts = [f"Current date and time: {now}", ""]

        system_parts.append(Prompt.PENNY_IDENTITY)

        if context:
            system_parts.append("")
            system_parts.append(context)

        if effective_prompt:
            if "{tools}" in effective_prompt:
                effective_prompt = effective_prompt.format(tools=self._build_tool_summary())
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

    def _build_tool_summary(self) -> str:
        """Build a dynamic tool summary from registered tools for prompt injection."""
        tools = self._tool_registry.get_all()
        if not tools:
            return ""
        names = [t.name for t in tools]
        logger.debug("Injecting tool summary into prompt: %s", ", ".join(names))
        lines = [f"- **{t.name}**: {t.description}" for t in tools]
        return "\n".join(lines)

    # ── Context building ─────────────────────────────────────────────────

    async def _build_context(
        self,
        user: str,
        content: str | None = None,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Build context text for system prompt and conversation history.

        Returns (context_text, conversation) where context_text is appended
        to the system prompt and conversation is user/assistant turns.
        """
        sections: list[str | None] = [
            self._build_profile_context(user, content),
            self._build_history_context(user),
            self._build_thought_context(user),
            self._build_dislike_context(user),
        ]
        context_text = "\n\n".join(s for s in sections if s)
        conversation = self._build_conversation(user)
        return context_text, conversation

    def _build_profile_context(self, sender: str, content: str | None) -> str | None:
        """Build user profile context string and configure search redaction."""
        try:
            user_info = self.db.users.get_info(sender)
            if not user_info:
                return None

            if content is not None:
                name = user_info.name
                user_said_name = bool(re.search(rf"\b{re.escape(name)}\b", content, re.IGNORECASE))
                if self._search_tool and isinstance(self._search_tool, SearchTool):
                    self._search_tool.redact_terms = [] if user_said_name else [name]

            logger.debug("Built profile context for %s", sender)
            return f"The user's name is {user_info.name}."
        except Exception:
            return None

    def _build_history_context(self, sender: str) -> str | None:
        """Build daily conversation history with dates and topic sub-bullets.

        Format:
            Mar 1:
            - topic1
            - topic2
            Today:
            - topic3
        """
        try:
            limit = int(self.config.runtime.HISTORY_CONTEXT_LIMIT)
            entries = self.db.history.get_recent(
                sender, PennyConstants.HistoryDuration.DAILY, limit=limit
            )
            if not entries:
                return None

            today = self._midnight_today()
            lines: list[str] = []
            for entry in entries:
                is_today = entry.period_start == today
                date_label = "Today" if is_today else entry.period_start.strftime("%b %-d")
                topics = self._extract_topic_lines(entry.topics)
                if topics:
                    lines.append(f"{date_label}:")
                    lines.extend(f"- {t}" for t in topics)
            if not lines:
                return None

            logger.debug("Built history context (%d entries)", len(entries))
            return "## Conversation History\n" + "\n".join(lines)
        except Exception:
            logger.warning("History context retrieval failed, proceeding without")
            return None

    @staticmethod
    def _extract_topic_lines(topics: str) -> list[str]:
        """Parse topic bullet text into clean topic strings."""
        result: list[str] = []
        for line in topics.strip().splitlines():
            topic = line.strip().lstrip("- ").strip()
            if topic:
                result.append(topic)
        return result

    def _build_thought_context(self, sender: str) -> str | None:
        """Build recent thinking summary context within freshness window."""
        try:
            hours = int(self.config.runtime.THOUGHT_FRESHNESS_HOURS)
            cutoff = self._freshness_cutoff(hours)
            all_thoughts = self.db.thoughts.get_recent(sender, limit=self.THOUGHT_CONTEXT_LIMIT)
            thoughts = [t for t in all_thoughts if t.created_at >= cutoff]
            if not thoughts:
                return None
            lines = [t.content for t in thoughts]
            logger.debug("Built thought context (%d thoughts)", len(thoughts))
            return "## Recent Background Thinking\n" + "\n\n".join(lines)
        except Exception:
            logger.warning("Thought context retrieval failed, proceeding without")
            return None

    def _build_dislike_context(self, user: str) -> str | None:
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
            return f"## Topics to Avoid\n{lines}"
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

    @staticmethod
    def _freshness_cutoff(hours: int) -> datetime:
        """Rolling cutoff: now minus N hours, as naive UTC."""
        return datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)

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
