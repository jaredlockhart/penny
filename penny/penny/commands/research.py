"""The /research command — start deep research on a topic."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import ResearchTask

logger = logging.getLogger(__name__)


class ResearchCommand(Command):
    """Start autonomous research on a topic."""

    name = "research"
    description = "Start deep research on a topic"
    help_text = (
        "Start an autonomous research task that performs multiple searches "
        "and produces a report.\n\n"
        "**Usage**: `/research <topic>`\n\n"
        "**Examples**:\n"
        "- `/research quantum computing applications`\n"
        "- `/research best coffee grinders 2026`\n\n"
        "Penny will run multiple search iterations in the background and "
        "post a comprehensive report when complete. You can continue the "
        "conversation while research runs."
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute research command."""
        topic = args.strip()

        if not topic:
            return CommandResult(
                text="Please specify a topic to research. Example: `/research quantum computing`"
            )

        # Get research config from runtime config (falls back to .env default)
        max_iterations = context.config.research_max_iterations

        # Use user ID as thread_id (works for Signal, Discord)
        thread_id = context.user

        # Check for existing in-progress research in this thread
        with Session(context.db.engine) as session:
            existing = session.exec(
                select(ResearchTask).where(
                    ResearchTask.thread_id == thread_id, ResearchTask.status == "in_progress"
                )
            ).first()

            if existing:
                return CommandResult(
                    text=f"Already researching '{existing.topic}' — please wait for that to finish"
                )

            # Create new research task
            task = ResearchTask(
                thread_id=thread_id,
                topic=topic,
                status="in_progress",
                max_iterations=max_iterations,
                created_at=datetime.now(UTC),
            )
            session.add(task)
            session.commit()
            logger.info("Created research task %d: %s", task.id, topic)

        return CommandResult(
            text=(
                f"Ok, started research on '{topic}'. "
                "I'll post results when done (this might take a few minutes)"
            )
        )
