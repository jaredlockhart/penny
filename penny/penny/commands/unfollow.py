"""/unfollow command — cancel a follow subscription."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse


class UnfollowCommand(Command):
    """Cancel a follow subscription by number."""

    name = "unfollow"
    description = "Stop following a topic"
    help_text = (
        "Cancel a follow subscription.\n\n"
        "**Usage**:\n"
        "• `/unfollow` — Show numbered list of active follows\n"
        "• `/unfollow <number>` — Cancel the follow at that position\n\n"
        "**Examples**:\n"
        "• `/unfollow`\n"
        "• `/unfollow 2`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Route to list or cancel based on args."""
        if not context.config.news_api_key:
            return CommandResult(text=PennyResponse.NEWS_NOT_CONFIGURED)

        args = args.strip()

        follows = context.db.follow_prompts.get_active(context.user)
        if not follows:
            return CommandResult(text=PennyResponse.FOLLOW_EMPTY)

        # No args — show numbered list
        if not args:
            return self._list_follows(follows)

        # Arg must be a number
        if not args.isdigit():
            return CommandResult(text=PennyResponse.FOLLOW_NOT_FOUND.format(number=args))

        return self._cancel_follow(int(args), follows, context)

    def _list_follows(self, follows: list) -> CommandResult:
        """Show numbered list of active follows."""
        lines = [PennyResponse.FOLLOW_LIST_HEADER, ""]
        for i, fp in enumerate(follows, 1):
            date = fp.created_at.strftime("%Y-%m-%d")
            lines.append(f"{i}. **{fp.prompt_text}** ({fp.cadence}) — since {date}")

        return CommandResult(text="\n".join(lines))

    def _cancel_follow(
        self, position: int, follows: list, context: CommandContext
    ) -> CommandResult:
        """Cancel follow at the given position."""
        if position < 1 or position > len(follows):
            return CommandResult(text=PennyResponse.FOLLOW_NOT_FOUND.format(number=position))

        fp = follows[position - 1]
        assert fp.id is not None
        context.db.follow_prompts.cancel(fp.id)
        return CommandResult(text=PennyResponse.FOLLOW_CANCELLED.format(topic=fp.prompt_text))
