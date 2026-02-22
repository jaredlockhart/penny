"""The /unlearn command — remove learn topics and their discovered entities."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse


class UnlearnCommand(Command):
    """Remove a learn topic and all entities discovered from it."""

    name = "unlearn"
    description = "Remove a learn topic and its discovered entities"
    help_text = (
        "Remove a past learn topic and delete all entities and facts discovered from it.\n\n"
        "**Usage**:\n"
        "- `/unlearn` — List all past learn topics (numbered, most recent first)\n"
        "- `/unlearn <number>` — Delete that topic and everything discovered from it\n\n"
        "**Examples**:\n"
        "- `/unlearn`\n"
        "- `/unlearn 3`"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute unlearn command."""
        args = args.strip()

        learn_prompts = context.db.get_user_learn_prompts(context.user)
        if not learn_prompts:
            return CommandResult(text=PennyResponse.UNLEARN_EMPTY)

        # No args — list all learn topics
        if not args:
            lines = [PennyResponse.UNLEARN_LIST_HEADER, ""]
            for i, lp in enumerate(learn_prompts, 1):
                status = f" *({lp.status})*" if lp.status == "active" else ""
                date = lp.created_at.strftime("%Y-%m-%d")
                lines.append(f"{i}. {lp.prompt_text}{status} — {date}")
            return CommandResult(text="\n".join(lines))

        # Arg must be a number
        if not args.isdigit():
            return CommandResult(text=PennyResponse.UNLEARN_INVALID_NUMBER.format(number=args))

        position = int(args)
        if position < 1 or position > len(learn_prompts):
            return CommandResult(text=PennyResponse.UNLEARN_INVALID_NUMBER.format(number=position))

        lp = learn_prompts[position - 1]
        assert lp.id is not None

        deleted_entities = context.db.delete_learn_prompt(lp.id)

        lines = [PennyResponse.UNLEARN_HEADER.format(topic=lp.prompt_text)]
        if deleted_entities:
            lines.append("")
            for name, fact_count in deleted_entities:
                lines.append(
                    PennyResponse.UNLEARN_ENTITY_LINE.format(name=name, fact_count=fact_count)
                )
        else:
            lines.append("")
            lines.append(PennyResponse.UNLEARN_NO_ENTITIES)

        return CommandResult(text="\n".join(lines))
