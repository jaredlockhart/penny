"""The /profile command — view or update user profile (name, location)."""

from __future__ import annotations

import logging
from datetime import datetime

try:
    from geopy.geocoders import Nominatim
    from timezonefinder import TimezoneFinder

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult

logger = logging.getLogger(__name__)


class ProfileCommand(Command):
    """View or update your basic user profile (name, location)."""

    name = "profile"
    description = "View or update your profile (name, location)"
    help_text = (
        "View your current profile or update your name and location.\n\n"
        "**Usage**:\n"
        "- `/profile` — View your current profile\n"
        "- `/profile <name> <location>` — Update your profile\n\n"
        "**Examples**:\n"
        "- `/profile Jared Toronto`\n"
        "- `/profile Seattle` (updates location only if name unchanged)\n\n"
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
                    text="You don't have a profile set up yet. Send me a message to get started!"
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

        # Parse update args - expecting "<name> <location>" or just "<location>"
        parts = args.split(maxsplit=1)
        if len(parts) == 0:
            return CommandResult(
                text="Usage: `/profile <name> <location>` or `/profile <location>`"
            )

        user_info = context.db.get_user_info(context.user)
        if not user_info:
            return CommandResult(
                text="You don't have a profile set up yet. Send me a message to get started!"
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
                text="Profile updates are not available (missing geopy/timezonefinder)."
            )

        timezone = await self._get_timezone(new_location)
        if not timezone:
            return CommandResult(
                text=f"I couldn't determine a timezone for '{new_location}'. "
                "Can you be more specific?"
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
            return CommandResult(text=f"Okay, I updated your {change_text}!")
        else:
            return CommandResult(text="Your profile is unchanged.")

    async def _get_timezone(self, location: str) -> str | None:
        """
        Derive IANA timezone from natural language location.

        Args:
            location: Natural language location

        Returns:
            IANA timezone string or None if lookup failed
        """
        try:
            geolocator = Nominatim(user_agent="penny_profile_command")
            geo_result = geolocator.geocode(location)
            if not geo_result:
                logger.warning("Geocoding failed for location: %s", location)
                return None

            tf = TimezoneFinder()
            timezone = tf.timezone_at(lat=geo_result.latitude, lng=geo_result.longitude)
            if not timezone:
                logger.warning("Timezone lookup failed for location: %s", location)
                return None

            logger.debug("Resolved timezone for %s: %s", location, timezone)
            return timezone

        except Exception as e:
            logger.warning("Timezone derivation failed for %s: %s", location, e)
            return None
