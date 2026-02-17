"""ResearchAgent for deep autonomous research on user-requested topics."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from penny.agents.base import Agent
from penny.agents.models import MessageRole, ToolCallRecord
from penny.config import Config
from penny.constants import RESEARCH_FOCUS_TIMEOUT_SECONDS
from penny.database.models import ResearchIteration, ResearchTask
from penny.prompts import (
    RESEARCH_FOLLOWUP_PROMPT,
    RESEARCH_REPORT_BUILD_PROMPT,
)
from penny.responses import PennyResponse
from penny.tools.builtin import SearchTool

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class ResearchAgent(Agent):
    """Agent for conducting autonomous multi-iteration research."""

    def __init__(self, config: Config, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None
        self._config = config
        self._pending_attachments: list[str] = []
        self._search_tool = self._find_search_tool()

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "research"

    def set_channel(self, channel: MessageChannel) -> None:
        """Set the channel for sending responses."""
        self._channel = channel

    async def execute(self) -> bool:
        """
        Execute next iteration of in-progress research task, or do nothing.

        Returns:
            True if work was done, False otherwise
        """
        if not self._channel:
            logger.error("ResearchAgent: no channel set")
            return False

        # Check for timed-out awaiting_focus tasks and auto-start them
        self._check_focus_timeout()

        # Find in-progress research task
        task = self._get_next_research_task()
        if not task:
            return False

        assert task.id is not None
        logger.info("Continuing research task %d: %s", task.id, task.topic)

        # Load previous iterations
        iterations = self._get_iterations(task.id)
        current_iteration = len(iterations)

        if current_iteration >= task.max_iterations:
            # Research complete - generate report
            await self._complete_research(task, iterations)
            return True

        # Run next search iteration (use followup prompt after first iteration)
        # Only fetch images on the last iteration (for the report attachment)
        is_last_iteration = current_iteration == task.max_iterations - 1
        if self._search_tool:
            self._search_tool.skip_images = not is_last_iteration

        history = self._build_history(task, iterations)
        prompt = (
            "Begin researching this topic." if current_iteration == 0 else RESEARCH_FOLLOWUP_PROMPT
        )
        response = await self.run(prompt=prompt, history=history)

        if not response.answer:
            logger.warning("Research iteration returned empty response")
            self._mark_failed(task.id, "Empty response from LLM")
            return False

        # Store attachments from last iteration for the report
        if is_last_iteration and response.attachments:
            self._pending_attachments = response.attachments

        # Extract the actual search query from tool calls (if any)
        search_query = self._extract_search_query(response.tool_calls)

        # Extract sources from raw response, then build/augment report
        sources = self._extract_sources(response.answer)
        current_report = iterations[-1].findings if iterations else None
        report_draft = await self._build_report(
            task.topic, response.answer, focus=task.focus, current_report=current_report
        )

        # Store iteration with the report draft
        # Sources stored separately per-iteration, merged at completion
        self._store_iteration(
            task_id=task.id,
            iteration_num=current_iteration + 1,
            query=search_query or f"Iteration {current_iteration + 1}",
            findings=report_draft,
            sources=sources,
        )

        logger.info(
            "Completed iteration %d/%d for task %d",
            current_iteration + 1,
            task.max_iterations,
            task.id,
        )

        return True

    async def _complete_research(
        self, task: ResearchTask, iterations: list[ResearchIteration]
    ) -> None:
        """Post the incrementally-built report to channel."""
        if not self._channel:
            return

        # The latest iteration's findings IS the report — append all unique sources
        report = iterations[-1].findings if iterations else ""
        all_sources: set[str] = set()
        for iteration in iterations:
            all_sources.update(json.loads(iteration.sources))
        if all_sources:
            report += "\n\n## sources\n"
            for source in sorted(all_sources):
                report += f"{source}\n"

        # Truncate if exceeds configured max length
        max_length = self._config.research_output_max_length
        if len(report) > max_length:
            truncate_at = max_length - len(PennyResponse.RESEARCH_TRUNCATED)
            report = report[:truncate_at] + PennyResponse.RESEARCH_TRUNCATED

        # Find recipient from thread
        recipient = self._find_recipient_for_thread(task.thread_id)
        if not recipient:
            logger.error("Could not find recipient for thread %s", task.thread_id)
            if task.id:
                self._mark_failed(task.id, "Could not find recipient")
            return

        # Send report with image attachment from last iteration (if any)
        attachments = self._pending_attachments or None
        self._pending_attachments = []
        typing_task = asyncio.create_task(self._channel._typing_loop(recipient))
        try:
            message_id = await self._channel.send_response(
                recipient,
                report,
                parent_id=None,  # Not continuing a specific thread
                attachments=attachments,
            )

            # Mark task complete and store message_id for continuation detection
            assert task.id is not None
            with Session(self.db.engine) as session:
                db_task = session.get(ResearchTask, task.id)
                if db_task:
                    db_task.status = "completed"
                    db_task.completed_at = datetime.now(UTC)
                    db_task.message_id = str(message_id) if message_id else None
                    session.add(db_task)
                    session.commit()
                    logger.info("Research task %d completed and marked in DB", task.id)

                    # Activate next pending task in this thread, if any
                    self._activate_next_pending_task(session, task.thread_id)
        finally:
            typing_task.cancel()
            await self._channel.send_typing(recipient, False)

    def _get_next_research_task(self) -> ResearchTask | None:
        """Find the next in-progress research task to work on."""
        with Session(self.db.engine) as session:
            return session.exec(
                select(ResearchTask)
                .where(ResearchTask.status == "in_progress")
                .order_by(ResearchTask.created_at.asc())  # type: ignore[unresolved-attribute]
            ).first()

    def _get_iterations(self, task_id: int) -> list[ResearchIteration]:
        """Load all iterations for a research task."""
        with Session(self.db.engine) as session:
            return list(
                session.exec(
                    select(ResearchIteration)
                    .where(ResearchIteration.research_task_id == task_id)
                    .order_by(ResearchIteration.iteration_num.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )

    def _build_history(
        self, task: ResearchTask, iterations: list[ResearchIteration]
    ) -> list[tuple[str, str]]:
        """Build conversation history for next iteration."""
        history: list[tuple[str, str]] = []

        # Add context about the research topic
        context = f"Research topic: {task.topic}"

        # If there's a parent task, mention it's a continuation
        if task.parent_task_id:
            context += "\nThis is a continuation of previous research on this topic."

        # Include user's focus to guide search direction
        if task.focus:
            context += f"\nUser's research focus: {task.focus}"

        # Include previous search queries so agent doesn't repeat them
        if iterations:
            queries = [it.query for it in iterations if it.query]
            if queries:
                context += "\nPrevious searches: " + ", ".join(f'"{q}"' for q in queries)

        history.append((MessageRole.SYSTEM.value, context))

        # Include the current report draft so agent knows what's been covered
        if iterations:
            history.append(
                (MessageRole.ASSISTANT.value, f"Current report draft:\n{iterations[-1].findings}")
            )

        return history

    def _store_iteration(
        self, task_id: int, iteration_num: int, query: str, findings: str, sources: list[str]
    ) -> None:
        """Store a research iteration in the database."""
        with Session(self.db.engine) as session:
            iteration = ResearchIteration(
                research_task_id=task_id,
                iteration_num=iteration_num,
                query=query,
                findings=findings,
                sources=json.dumps(sources),
                timestamp=datetime.now(UTC),
            )
            session.add(iteration)
            session.commit()

    def _extract_search_query(self, tool_calls: list[ToolCallRecord]) -> str | None:
        """Extract the search query from tool call records."""
        for tc in tool_calls:
            if tc.tool == "search" and "query" in tc.arguments:
                return tc.arguments["query"]
        return None

    def _extract_sources(self, content: str) -> list[str]:
        """Extract URLs from response content."""
        sources = []
        for line in content.split("\n"):
            if line.startswith("http://") or line.startswith("https://"):
                sources.append(line.strip())
        return sources

    async def _build_report(
        self,
        topic: str,
        raw_response: str,
        focus: str | None = None,
        current_report: str | None = None,
    ) -> str:
        """Build or augment the research report from new search results."""
        user_content = f"Research topic: {topic}"
        if focus:
            user_content += f"\nRequested report format: {focus}"
        if current_report:
            user_content += f"\n\nExisting report draft:\n\n{current_report}"
        user_content += f"\n\nNew search results:\n\n{raw_response}"

        response = await self._ollama_client.chat(
            messages=[
                {"role": "system", "content": RESEARCH_REPORT_BUILD_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        return response.message.content.strip()

    def _mark_failed(self, task_id: int, reason: str) -> None:
        """Mark a research task as failed."""
        with Session(self.db.engine) as session:
            task = session.get(ResearchTask, task_id)
            if task:
                task.status = "failed"
                task.completed_at = datetime.now(UTC)
                session.add(task)
                session.commit()
                logger.error("Research task %d marked as failed: %s", task_id, reason)

                # Activate next pending task in this thread, if any
                self._activate_next_pending_task(session, task.thread_id)

    def _activate_next_pending_task(self, session: Session, thread_id: str) -> None:
        """Activate the next pending research task in a thread."""
        next_task = session.exec(
            select(ResearchTask)
            .where(ResearchTask.thread_id == thread_id, ResearchTask.status == "pending")
            .order_by(ResearchTask.created_at.asc())  # type: ignore[unresolved-attribute]
        ).first()

        if next_task:
            next_task.status = "in_progress"
            session.add(next_task)
            session.commit()
            logger.info("Activated pending research task %d: %s", next_task.id, next_task.topic)

    def _check_focus_timeout(self) -> None:
        """Auto-start awaiting_focus tasks that have exceeded the timeout."""
        with Session(self.db.engine) as session:
            tasks = session.exec(
                select(ResearchTask).where(ResearchTask.status == "awaiting_focus")
            ).all()

            now = datetime.now(UTC)
            for task in tasks:
                # SQLite round-trip may strip tzinfo; normalize to UTC-aware
                created = (
                    task.created_at.replace(tzinfo=UTC)
                    if task.created_at.tzinfo is None
                    else task.created_at
                )
                elapsed = (now - created).total_seconds()
                if elapsed >= RESEARCH_FOCUS_TIMEOUT_SECONDS:
                    task.status = "in_progress"
                    session.add(task)
                    logger.info(
                        "Research task %d auto-started after focus timeout (%ds)",
                        task.id,
                        int(elapsed),
                    )
            session.commit()

    def _find_search_tool(self) -> SearchTool | None:
        """Find the SearchTool instance in this agent's tools."""
        for tool in self.tools:
            if isinstance(tool, SearchTool):
                return tool
        return None

    def _find_recipient_for_thread(self, thread_id: str) -> str | None:
        """Find a recipient (user) who has sent messages in this thread."""
        # For Signal, thread_id is the user's phone number
        # For Discord, thread_id is the channel ID — we just return it
        # This is a simplified approach; in practice we'd look up recent messages
        return thread_id
