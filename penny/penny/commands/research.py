"""The /research command — start deep research on a topic."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, func, select

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import ResearchIteration, ResearchTask
from penny.prompts import RESEARCH_OUTPUT_OPTIONS_PROMPT, RESEARCH_OUTPUT_OPTIONS_SYSTEM_PROMPT

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
        "- `/research ! best coffee grinders 2026` — skip clarification\n"
        "- `/research cancel 2` — cancel research task #2\n\n"
        "Penny will suggest output formats before starting research. "
        "Reply with a number, describe what you want, or say 'go' to start immediately.\n"
        "Use `!` prefix to skip the output format step and start research right away.\n\n"
        "You can continue the conversation while research runs."
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute research command."""
        topic = args.strip()

        # If no topic provided, list active, pending, and awaiting_focus tasks
        if not topic:
            return self._list_tasks(context)

        # Handle cancel subcommand
        if topic.startswith("cancel"):
            return self._cancel_task(topic[6:].strip(), context)

        # Check for skip-clarification prefix
        skip_clarification = topic.startswith("! ")
        if skip_clarification:
            topic = topic[2:].strip()
            if not topic:
                return CommandResult(text="Please provide a topic after `!`")

        # Get research config from runtime config (falls back to .env default)
        max_iterations = context.config.research_max_iterations

        # Use user ID as thread_id (works for Signal, Discord)
        thread_id = context.user

        # Check for existing in-progress or awaiting_focus research in this thread
        with Session(context.db.engine) as session:
            existing = session.exec(
                select(ResearchTask).where(
                    ResearchTask.thread_id == thread_id,
                    ResearchTask.status.in_(["in_progress", "awaiting_focus"]),  # type: ignore[unresolved-attribute]
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

            if skip_clarification:
                # Create task directly as in_progress (no clarification)
                task = ResearchTask(
                    thread_id=thread_id,
                    topic=topic,
                    status="in_progress",
                    max_iterations=max_iterations,
                    created_at=datetime.now(UTC),
                )
                session.add(task)
                session.commit()
                logger.info("Created research task %d (skip clarification): %s", task.id, topic)

                return CommandResult(
                    text=(
                        f"Ok, started research on '{topic}'. "
                        "I'll post results when done (this might take a few minutes)"
                    )
                )

            # Default: generate output format options and create task as awaiting_focus
            options = await self._generate_output_options(topic, context)

            task = ResearchTask(
                thread_id=thread_id,
                topic=topic,
                status="awaiting_focus",
                options=options,
                max_iterations=max_iterations,
                created_at=datetime.now(UTC),
            )
            session.add(task)
            session.commit()
            logger.info("Created research task %d (awaiting focus): %s", task.id, topic)

        return CommandResult(
            text=(
                f"What should the report focus on for '{topic}'?\n\n"
                f"{options}\n\n"
                "Reply with a number, describe what you want, or say 'go' to start right away."
            )
        )

    def _list_tasks(self, context: CommandContext) -> CommandResult:
        """List active, pending, and awaiting_focus research tasks."""
        with Session(context.db.engine) as session:
            tasks = session.exec(
                select(ResearchTask)
                .where(
                    ResearchTask.thread_id == context.user,
                    ResearchTask.status.in_(["in_progress", "pending", "awaiting_focus"]),  # type: ignore[unresolved-attribute]
                )
                .order_by(ResearchTask.created_at.asc())  # type: ignore[unresolved-attribute]
            ).all()

            if not tasks:
                return CommandResult(text="No active research tasks")

            lines = ["**Currently researching:**\n"]
            for i, task in enumerate(tasks, 1):
                iteration_count = session.exec(
                    select(func.count(ResearchIteration.id)).where(
                        ResearchIteration.research_task_id == task.id
                    )
                ).one()

                if task.status == "pending":
                    progress = "*Queued*"
                elif task.status == "awaiting_focus":
                    progress = "*Awaiting focus*"
                else:
                    progress = f"{iteration_count}/{task.max_iterations}"

                lines.append(f"{i}. {task.topic} — {progress}")

            lines.append("\nCancel with `/research cancel <number>`")

            return CommandResult(text="\n".join(lines))

    def _cancel_task(self, number_str: str, context: CommandContext) -> CommandResult:
        """Cancel a research task by its list position."""
        if not number_str:
            return CommandResult(text="Please provide a task number: `/research cancel 1`")

        try:
            position = int(number_str)
        except ValueError:
            return CommandResult(text=f"'{number_str}' is not a valid number")

        if position < 1:
            return CommandResult(text="Task number must be 1 or higher")

        with Session(context.db.engine) as session:
            tasks = list(
                session.exec(
                    select(ResearchTask)
                    .where(
                        ResearchTask.thread_id == context.user,
                        ResearchTask.status.in_(["in_progress", "pending", "awaiting_focus"]),  # type: ignore[unresolved-attribute]
                    )
                    .order_by(ResearchTask.created_at.asc())  # type: ignore[unresolved-attribute]
                ).all()
            )

            if not tasks:
                return CommandResult(text="No active research tasks to cancel")

            if position > len(tasks):
                return CommandResult(
                    text=f"Only {len(tasks)} active task(s) — pick a number from 1 to {len(tasks)}"
                )

            task = tasks[position - 1]
            topic = task.topic
            task.status = "cancelled"
            session.add(task)
            session.commit()
            logger.info("Cancelled research task %d: %s", task.id, topic)

            return CommandResult(text=f"Cancelled research on '{topic}'")

    async def _generate_output_options(self, topic: str, context: CommandContext) -> str:
        """Generate output format options for a research topic using LLM."""
        prompt = RESEARCH_OUTPUT_OPTIONS_PROMPT.format(topic=topic)
        response = await context.ollama_client.chat(
            messages=[
                {"role": "system", "content": RESEARCH_OUTPUT_OPTIONS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return response.message.content.strip()
