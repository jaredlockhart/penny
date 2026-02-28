"""/events command — view recent events discovered by Penny."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.database.models import Event
from penny.responses import PennyResponse


class EventsCommand(Command):
    """View recent events discovered from followed topics."""

    name = "events"
    description = "View recent events"
    help_text = (
        "View events Penny has discovered from your followed topics.\n\n"
        "**Usage**:\n"
        "• `/events` — List recent events (last 7 days)\n"
        "• `/events <number>` — Show full details for an event\n\n"
        "**Examples**:\n"
        "• `/events`\n"
        "• `/events 3`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Route to list or detail based on args."""
        if not context.config.news_api_key:
            return CommandResult(text=PennyResponse.NEWS_NOT_CONFIGURED)

        args = args.strip()

        events = context.db.events.get_recent(context.user)
        if not events:
            return CommandResult(text=PennyResponse.EVENTS_EMPTY)

        if not args:
            return self._list_events(events)

        if not args.isdigit():
            return CommandResult(text=PennyResponse.EVENTS_NOT_FOUND.format(number=args))

        return self._show_detail(int(args), events, context)

    def _list_events(self, events: list[Event]) -> CommandResult:
        """Show recent events as a numbered list."""
        lines = [PennyResponse.EVENTS_LIST_HEADER, ""]
        for i, event in enumerate(events, 1):
            date = event.occurred_at.strftime("%Y-%m-%d")
            lines.append(f"{i}. **{event.headline}** — {date}")

        return CommandResult(text="\n".join(lines))

    def _show_detail(
        self, position: int, events: list[Event], context: CommandContext
    ) -> CommandResult:
        """Show full detail for an event at the given position."""
        if position < 1 or position > len(events):
            return CommandResult(text=PennyResponse.EVENTS_NOT_FOUND.format(number=position))

        event = events[position - 1]
        assert event.id is not None
        lines = [f"**{event.headline}**"]

        date = event.occurred_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"**Date**: {date}")

        if event.summary:
            lines.append("")
            lines.append(event.summary)

        if event.source_url:
            lines.append("")
            lines.append(event.source_url)

        return CommandResult(text="\n".join(lines))
