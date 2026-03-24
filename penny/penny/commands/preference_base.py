"""Base classes for preference add and remove commands."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.constants import PennyConstants
from penny.database.models import Preference
from penny.responses import PennyResponse


class PreferenceBaseCommand(Command):
    """Shared helpers for preference commands."""

    valence: str  # Set by subclasses

    def _empty_message(self) -> str:
        if self.valence == PennyConstants.PreferenceValence.POSITIVE:
            return PennyResponse.PREF_NO_LIKES
        return PennyResponse.PREF_NO_DISLIKES

    def _header(self) -> str:
        if self.valence == PennyConstants.PreferenceValence.POSITIVE:
            return PennyResponse.PREF_LIKES_HEADER
        return PennyResponse.PREF_DISLIKES_HEADER

    def _valence_label(self) -> str:
        if self.valence == PennyConstants.PreferenceValence.POSITIVE:
            return "like"
        return "dislike"

    def _list_preferences(self, prefs: list[Preference]) -> CommandResult:
        lines = [self._header(), ""]
        for idx, pref in enumerate(prefs, start=1):
            lines.append(f"{idx}. {pref.content}")
        return CommandResult(text="\n".join(lines))


class PreferenceAddCommand(PreferenceBaseCommand):
    """Base for /like, /dislike — list preferences or add a new one."""

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        args = args.strip()
        prefs = context.db.preferences.get_for_user_by_valence(context.user, self.valence)

        if not args:
            if not prefs:
                return CommandResult(text=self._empty_message())
            return self._list_preferences(prefs)

        return self._add_preference(args, context)

    def _add_preference(self, content: str, context: CommandContext) -> CommandResult:
        context.db.preferences.add(
            user=context.user,
            content=content,
            valence=self.valence,
            source=PennyConstants.PreferenceSource.MANUAL,
        )
        label = self._valence_label()
        return CommandResult(text=PennyResponse.PREF_ADDED.format(content=content, valence=label))


class PreferenceRemoveCommand(PreferenceBaseCommand):
    """Base for /unlike, /undislike — list preferences or remove one by number."""

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        args = args.strip()
        prefs = context.db.preferences.get_for_user_by_valence(context.user, self.valence)

        if not prefs:
            return CommandResult(text=self._empty_message())

        if not args:
            return self._list_preferences(prefs)

        if not args.isdigit():
            return CommandResult(text=PennyResponse.PREF_INVALID_NUMBER.format(number=args))

        return self._delete_preference(int(args), prefs, context)

    def _delete_preference(
        self, position: int, prefs: list[Preference], context: CommandContext
    ) -> CommandResult:
        if position < 1 or position > len(prefs):
            return CommandResult(
                text=PennyResponse.PREF_NO_PREF_WITH_NUMBER.format(number=position)
            )

        to_delete = prefs[position - 1]
        context.db.preferences.delete(to_delete.id)  # type: ignore[arg-type]

        remaining = [p for p in prefs if p.id != to_delete.id]
        label = self._valence_label()
        deleted_msg = PennyResponse.PREF_DELETED.format(content=to_delete.content, valence=label)

        if not remaining:
            return CommandResult(
                text=f"{deleted_msg}\n\n{PennyResponse.PREF_DELETED_NO_REMAINING.format(valence=label)}"
            )

        lines = [f"{deleted_msg}\n", PennyResponse.PREF_STILL_REMAINING]
        for idx, pref in enumerate(remaining, start=1):
            lines.append(f"{idx}. {pref.content}")
        return CommandResult(text="\n".join(lines))
