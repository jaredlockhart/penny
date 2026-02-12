"""ResearchAgent for deep autonomous research on user-requested topics."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from penny.agents.base import Agent
from penny.agents.models import MessageRole
from penny.config import Config
from penny.constants import RESEARCH_PROMPT
from penny.database.models import ResearchIteration, ResearchTask

if TYPE_CHECKING:
    from penny.channels import MessageChannel

logger = logging.getLogger(__name__)


class ResearchAgent(Agent):
    """Agent for conducting autonomous multi-iteration research."""

    def __init__(self, config: Config, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._channel: MessageChannel | None = None
        self._config = config

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

        # Run next search iteration
        history = self._build_history(task, iterations)
        response = await self.run(prompt=RESEARCH_PROMPT, history=history)

        if not response.answer:
            logger.warning("Research iteration returned empty response")
            self._mark_failed(task.id, "Empty response from LLM")
            return False

        # Extract findings and sources from response
        findings = response.answer
        sources = self._extract_sources(response.answer)

        # Store iteration
        self._store_iteration(
            task_id=task.id,
            iteration_num=current_iteration + 1,
            query=f"Iteration {current_iteration + 1}",  # Query inferred by model
            findings=findings,
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
        """Generate final report and post to channel."""
        if not self._channel:
            return

        # Generate report
        report = self._generate_report(task.topic, iterations)

        # Truncate if exceeds configured max length
        max_length = self._config.research_output_max_length
        truncation_message = (
            "\n\n[Report truncated due to length limits — reply to request full details]"
        )
        if len(report) > max_length:
            truncate_at = max_length - len(truncation_message)
            report = report[:truncate_at] + truncation_message

        # Find recipient from thread
        recipient = self._find_recipient_for_thread(task.thread_id)
        if not recipient:
            logger.error("Could not find recipient for thread %s", task.thread_id)
            if task.id:
                self._mark_failed(task.id, "Could not find recipient")
            return

        # Send report
        typing_task = asyncio.create_task(self._channel._typing_loop(recipient))
        try:
            message_id = await self._channel.send_response(
                recipient,
                report,
                parent_id=None,  # Not continuing a specific thread
                attachments=None,
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

        history.append((MessageRole.SYSTEM.value, context))

        # Add previous iterations as assistant responses
        for iteration in iterations:
            history.append(
                (
                    MessageRole.ASSISTANT.value,
                    f"Search findings (iteration {iteration.iteration_num}):\n{iteration.findings}",
                )
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

    def _extract_sources(self, content: str) -> list[str]:
        """Extract URLs from response content."""
        sources = []
        for line in content.split("\n"):
            if line.startswith("http://") or line.startswith("https://"):
                sources.append(line.strip())
        return sources

    def _generate_report(self, topic: str, iterations: list[ResearchIteration]) -> str:
        """Generate final markdown report from all iterations."""
        # Collect all findings and deduplicate sources
        all_findings: list[str] = []
        all_sources: set[str] = set()

        for iteration in iterations:
            all_findings.append(iteration.findings)
            sources_list = json.loads(iteration.sources)
            all_sources.update(sources_list)

        # Build report
        report_lines = [
            f"research complete: {topic}",
            "",
            "## summary",
            self._summarize_findings(all_findings),
            "",
            "## key findings",
        ]

        # Extract bullet points from findings
        for finding in all_findings:
            # Look for bullet points in the findings
            for line in finding.split("\n"):
                stripped = line.strip()
                # Skip markdown headers (e.g., ## TL;DR)
                if stripped.startswith("#"):
                    continue
                # Skip ALL markdown table rows (any line starting with |)
                if stripped.startswith("|"):
                    continue
                if stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("*"):
                    report_lines.append(stripped)
                elif stripped and not stripped.startswith("http"):
                    # Add non-URL content as bullet if it's not already bulletted
                    report_lines.append(f"• {stripped}")

        report_lines.append("")
        report_lines.append("## sources")
        for source in sorted(all_sources):
            report_lines.append(source)

        return "\n".join(report_lines)

    def _summarize_findings(self, findings: list[str]) -> str:
        """Generate 2-3 sentence summary from all findings."""
        # Filter out markdown headers and tables from findings before summarizing
        cleaned_findings = []
        for finding in findings[:3]:
            # Remove markdown headers and table lines
            cleaned_lines = []
            for line in finding.split("\n"):
                stripped = line.strip()
                # Skip markdown headers
                if stripped.startswith("#"):
                    continue
                # Skip ALL markdown table rows (any line starting with |)
                if stripped.startswith("|"):
                    continue
                if stripped:
                    cleaned_lines.append(stripped)
            if cleaned_lines:
                cleaned_findings.append(" ".join(cleaned_lines))

        # Take first sentence from each cleaned finding
        summary_sentences = []
        for cleaned in cleaned_findings:
            # Get first sentence (split on period followed by space or newline)
            first_sentence = cleaned.split(". ")[0]
            if first_sentence:
                summary_sentences.append(first_sentence.strip())

        return " ".join(summary_sentences)[:300]  # Cap at 300 chars

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

    def _find_recipient_for_thread(self, thread_id: str) -> str | None:
        """Find a recipient (user) who has sent messages in this thread."""
        # For Signal, thread_id is the user's phone number
        # For Discord, thread_id is the channel ID — we just return it
        # This is a simplified approach; in practice we'd look up recent messages
        return thread_id
