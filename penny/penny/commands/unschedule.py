"""/unschedule command — delete a recurring scheduled task."""

from __future__ import annotations

from sqlmodel import Session, select

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Schedule
from penny.responses import PennyResponse


class UnscheduleCommand(Command):
    """Delete a recurring scheduled task by number."""

    name = "unschedule"
    description = "Delete a scheduled task"
    help_text = (
        "Delete a recurring scheduled task.\n\n"
        "**Usage**:\n"
        "• `/unschedule` — Show numbered list of active schedules\n"
        "• `/unschedule <number>` — Delete the schedule at that position\n\n"
        "**Examples**:\n"
        "• `/unschedule`\n"
        "• `/unschedule 2`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Route to list or delete based on args."""
        args = args.strip()

        with Session(context.db.engine) as session:
            schedules = list(
                session.exec(
                    select(Schedule).where(Schedule.user_id == context.user).order_by(Schedule.id)  # type: ignore[arg-type]
                )
            )

            if not schedules:
                return CommandResult(text=PennyResponse.SCHEDULE_NO_TASKS)

            # No args — show numbered list
            if not args:
                return self._list_schedules(schedules)

            # Arg must be a number
            if not args.isdigit():
                return CommandResult(text=PennyResponse.SCHEDULE_INVALID_NUMBER.format(number=args))

            return self._delete_schedule(int(args), schedules, session)

    def _list_schedules(self, schedules: list[Schedule]) -> CommandResult:
        """Show numbered list of active schedules."""
        lines = ["**Your Schedules**", ""]
        for idx, sched in enumerate(schedules, start=1):
            lines.append(f"{idx}. **{sched.timing_description}**: {sched.prompt_text}")

        return CommandResult(text="\n".join(lines))

    def _delete_schedule(
        self, position: int, schedules: list[Schedule], session: Session
    ) -> CommandResult:
        """Delete schedule at the given position."""
        if position < 1 or position > len(schedules):
            return CommandResult(
                text=PennyResponse.SCHEDULE_NO_SCHEDULE_WITH_NUMBER.format(number=position)
            )

        to_delete = schedules[position - 1]
        session.delete(to_delete)
        session.commit()

        remaining = [s for s in schedules if s.id != to_delete.id]
        deleted_msg = PennyResponse.SCHEDULE_DELETED_PREFIX.format(
            timing=to_delete.timing_description, prompt=to_delete.prompt_text
        )

        if not remaining:
            return CommandResult(
                text=f"{deleted_msg}\n\n{PennyResponse.SCHEDULE_DELETED_NO_REMAINING}"
            )

        lines = [f"{deleted_msg}\n", PennyResponse.SCHEDULE_STILL_SCHEDULED]
        for idx, sched in enumerate(remaining, start=1):
            lines.append(f"{idx}. **{sched.timing_description}**: {sched.prompt_text}")

        return CommandResult(text="\n".join(lines))
