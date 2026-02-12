"""The /research command — start deep research on a topic."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, func, select

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import ResearchIteration, ResearchTask

logger = logging.getLogger(__name__)


class ResearchCommand(Command):
    """Start autonomous research on a topic."""

    name = "research"
    description = "Start deep research on a topic"
    help_text = (
        "Start an autonomous research task that performs multiple searches "
        "and produces a report.\n\n"
        "**Usage**: `/research [topic]`\n\n"
        "**Examples**:\n"
        "- `/research` — list active research tasks\n"
        "- `/research quantum computing applications`\n"
        "- `/research best coffee grinders 2026`\n\n"
        "Penny will run multiple search iterations in the background and "
        "post a comprehensive report when complete. You can continue the "
        "conversation while research runs."
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute research command."""
        topic = args.strip()

        # If no topic provided, list active and pending research tasks
        if not topic:
            with Session(context.db.engine) as session:
                # Get all in_progress and pending tasks for this user/thread
                tasks = session.exec(
                    select(ResearchTask)
                    .where(
                        ResearchTask.thread_id == context.user,
                        ResearchTask.status.in_(["in_progress", "pending"]),  # type: ignore[unresolved-attribute]
                    )
                    .order_by(ResearchTask.created_at.asc())  # type: ignore[unresolved-attribute]
                ).all()

                if not tasks:
                    return CommandResult(text="No active research tasks")

                # Format task list with progress
                lines = ["**Currently researching:**\n"]
                for task in tasks:
                    # Count completed iterations for this task
                    iteration_count = session.exec(
                        select(func.count(ResearchIteration.id)).where(
                            ResearchIteration.research_task_id == task.id
                        )
                    ).one()

                    # Format progress: "7/10", "*Not Started*", or "*Queued*"
                    if task.status == "pending":
                        progress = "*Queued*"
                    elif iteration_count == 0:
                        progress = "*Not Started*"
                    else:
                        progress = f"{iteration_count}/{task.max_iterations}"

                    lines.append(f"* {task.topic} {progress}")

                return CommandResult(text="\n".join(lines))

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

            # If there's an active task, queue the new one as pending
            if existing:
                task = ResearchTask(
                    thread_id=thread_id,
                    topic=topic,
                    status="pending",
                    max_iterations=max_iterations,
                    created_at=datetime.now(UTC),
                )
                session.add(task)
                session.commit()
                logger.info("Queued research task %d: %s", task.id, topic)
                return CommandResult(
                    text=f"Queued '{topic}' for research (currently researching '{existing.topic}')"
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
