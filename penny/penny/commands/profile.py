"""The /profile command — view or update user profile (name, location, DOB)."""

from __future__ import annotations

import logging
from datetime import datetime

try:
    from penny.profile.utils import get_timezone

    HAS_GEO = True
except ImportError:
    get_timezone = None  # type: ignore[assignment]
    HAS_GEO = False

try:
    import dateparser

    HAS_DATEPARSER = True
except ImportError:
    dateparser = None  # type: ignore[assignment]
    HAS_DATEPARSER = False

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult

logger = logging.getLogger(__name__)


class ProfileCommand(Command):
    """View or update your basic user profile (name, location, date of birth)."""

    name = "profile"
    description = "View or update your profile (name, location, DOB)"
    help_text = (
        "View your current profile or create/update your profile information.\n\n"
        "**Usage**:\n"
        "- `/profile` — View your current profile\n"
        "- `/profile <name> <location> <date of birth>` — Create profile (if new)\n"
        "- `/profile <name> <location>` — Update name/location (if profile exists)\n\n"
        "**Examples**:\n"
        "- `/profile alex seattle january 10 1995` (initial setup)\n"
        "- `/profile jared toronto` (update existing)\n"
        "- `/profile seattle` (update location only)\n\n"
        "**Note**: Timezone is automatically derived from your location."
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute profile command."""

        args = args.strip()

        # No args - show current profile
        if not args:
            user_info = context.db.get_user_info(context.user)
            if not user_info:
                return CommandResult(
                    text=(
                        "you don't have a profile yet! set it up with:\n"
                        "`/profile <name> <location> <date of birth>`\n\n"
                        "for example: `/profile alex seattle january 10 1995` 📝"
                    )
                )

            # Format date of birth for display
            dob_formatted = datetime.strptime(user_info.date_of_birth, "%Y-%m-%d").strftime(
                "%B %d, %Y"
            )

            lines = [
                "**Your Profile**",
                "",
                f"**Name**: {user_info.name}",
                f"**Location**: {user_info.location}",
                f"**Timezone**: {user_info.timezone}",
                f"**Date of Birth**: {dob_formatted}",
            ]
            return CommandResult(text="\n".join(lines))

        user_info = context.db.get_user_info(context.user)

        # NEW PROFILE CREATION (no existing profile)
        if not user_info:
            if not HAS_GEO or not HAS_DATEPARSER:
                return CommandResult(
                    text="profile creation not available (missing dependencies) 😞"
                )

            # Parse args as "<name> <location> <dob>"
            # We need at least 3 tokens (name, location, dob-start)
            parts = args.split(maxsplit=2)
            if len(parts) < 3:
                return CommandResult(
                    text=(
                        "to create your profile, i need name, location, and date of birth.\n\n"
                        "usage: `/profile <name> <location> <date of birth>`\n"
                        "example: `/profile alex seattle january 10 1995`"
                    )
                )

            new_name = parts[0].strip()
            new_location = parts[1].strip()
            dob_text = parts[2].strip()

            # Parse date of birth
            dob_date = dateparser.parse(dob_text, settings={"PREFER_DATES_FROM": "past"})
            if not dob_date:
                return CommandResult(
                    text=(
                        f"i couldn't parse '{dob_text}' as a date. "
                        "try something like 'january 10 1995' 📅"
                    )
                )

            dob_formatted = dob_date.strftime("%Y-%m-%d")

            # Derive timezone from location
            timezone = await get_timezone(new_location)
            if not timezone:
                return CommandResult(
                    text=(
                        f"i couldn't find a timezone for '{new_location}'. "
                        "can you be more specific? 🗺️"
                    )
                )

            # Save new profile
            context.db.save_user_info(
                sender=context.user,
                name=new_name,
                location=new_location,
                timezone=timezone,
                date_of_birth=dob_formatted,
            )

            return CommandResult(text=f"got it! your profile is set up. welcome, {new_name}! 🎉")

        # PROFILE UPDATE (existing profile)
        parts = args.split(maxsplit=1)
        if len(parts) == 0:
            return CommandResult(
                text="usage: `/profile <name> <location>` or `/profile <location>`"
            )

        # If two parts, first is name, second is location
        # If one part, it's location (keep existing name)
        if len(parts) == 2:
            new_name = parts[0].strip()
            new_location = parts[1].strip()
        else:
            new_name = user_info.name
            new_location = parts[0].strip()

        # Derive new timezone from location
        if not HAS_GEO:
            return CommandResult(
                text="profile updates not available (missing geopy/timezonefinder) 😞"
            )

        timezone = await get_timezone(new_location)
        if not timezone:
            return CommandResult(
                text=f"i couldn't find a timezone for '{new_location}'. can you be more specific? 🗺️"
            )

        # Update database
        context.db.save_user_info(
            sender=context.user,
            name=new_name,
            location=new_location,
            timezone=timezone,
            date_of_birth=user_info.date_of_birth,  # Keep existing DOB
        )

        # Build confirmation message
        changes = []
        if new_name != user_info.name:
            changes.append(f"name to **{new_name}**")
        if new_location != user_info.location:
            changes.append(f"location to **{new_location}** ({timezone})")

        if changes:
            change_text = " and ".join(changes)
            return CommandResult(text=f"okay, i updated your {change_text}! ✅")
        else:
            return CommandResult(text="your profile is unchanged 🤷")
