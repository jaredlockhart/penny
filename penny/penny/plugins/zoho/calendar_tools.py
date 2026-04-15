"""Zoho Calendar tools — LLM-callable tools for calendar management."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.plugins.zoho.calendar_client import ZohoCalendarClient

logger = logging.getLogger(__name__)


class ListCalendarsArgs(BaseModel):
    """Arguments for listing calendars."""

    pass


class GetEventsArgs(BaseModel):
    """Arguments for getting calendar events."""

    calendar_name: str | None = Field(
        default=None, description="Calendar name (uses 'Default' if not specified)"
    )
    days_ahead: int = Field(default=14, description="Number of days ahead to search")


class CheckAvailabilityArgs(BaseModel):
    """Arguments for checking availability."""

    start_date: str = Field(description="Start date/time (ISO format or natural language)")
    end_date: str = Field(description="End date/time (ISO format or natural language)")
    attendees: list[str] | None = Field(
        default=None, description="Optional list of attendee emails to check"
    )


class CreateEventArgs(BaseModel):
    """Arguments for creating a calendar event."""

    title: str = Field(description="Event title")
    start: str = Field(description="Start date/time (ISO format)")
    end: str = Field(description="End date/time (ISO format)")
    calendar_name: str | None = Field(
        default=None, description="Calendar name (uses 'Default' if not specified)"
    )
    description: str | None = Field(default=None, description="Event description")
    location: str | None = Field(default=None, description="Event location")
    attendees: list[str] | None = Field(
        default=None, description="List of attendee email addresses"
    )
    is_allday: bool = Field(default=False, description="Whether this is an all-day event")


class FindFreeSlotsArgs(BaseModel):
    """Arguments for finding free time slots."""

    duration_minutes: int = Field(description="Required slot duration in minutes")
    days_ahead: int = Field(default=14, description="Number of days ahead to search")
    attendees: list[str] | None = Field(
        default=None, description="Optional attendee emails to consider"
    )


class UpdateEventArgs(BaseModel):
    """Arguments for updating a calendar event."""

    event_title: str = Field(description="Title of the event to update (for searching)")
    calendar_name: str | None = Field(
        default=None, description="Calendar name where the event exists"
    )
    new_title: str | None = Field(default=None, description="New event title")
    new_start: str | None = Field(default=None, description="New start date/time (ISO format)")
    new_end: str | None = Field(default=None, description="New end date/time (ISO format)")
    new_description: str | None = Field(default=None, description="New event description")
    new_location: str | None = Field(default=None, description="New event location")
    recurrence_edittype: str = Field(
        default="all",
        description="""For recurring events: 'all' (all occurrences),
        'following' (this and future), 'only' (just this one)""",
    )


class ListCalendarsTool(Tool):
    """List available calendars."""

    name = "list_calendars"
    description = (
        "List all available calendars in the user's Zoho Calendar account. "
        "Returns calendar names, colors, and timezones. Use this to discover "
        "what calendars exist before creating events or checking availability."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing calendars"

    def __init__(self, calendar_client: ZohoCalendarClient) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> str:
        """List all calendars."""
        calendars = await self._client.get_calendars()
        if not calendars:
            return "No calendars found."

        lines = [f"Found {len(calendars)} calendar(s):\n"]
        for cal in calendars:
            default_marker = " (default)" if cal.is_default else ""
            lines.append(f"- **{cal.name}**{default_marker}")
            if cal.timezone:
                lines.append(f"  Timezone: {cal.timezone}")
        return "\n".join(lines)


class GetEventsTool(Tool):
    """Get upcoming calendar events."""

    name = "get_events"
    description = (
        "Get upcoming events from a calendar. Returns event titles, times, "
        "locations, and attendees. Use this to check what's scheduled or "
        "to find conflicts before scheduling new events."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "calendar_name": {
                "type": "string",
                "description": (
                    "Calendar name to get events from. Uses 'Default' if not specified. "
                    "Examples: 'Default', 'Studio A', 'Personal'"
                ),
            },
            "days_ahead": {
                "type": "integer",
                "description": "Number of days ahead to search (default: 14)",
            },
        },
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Getting calendar events"

    def __init__(self, calendar_client: ZohoCalendarClient) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> str:
        """Get events from a calendar."""
        args = GetEventsArgs(**kwargs)

        # Find the calendar
        if args.calendar_name:
            calendar = await self._client.get_calendar_by_name(args.calendar_name)
            if not calendar:
                return f"Calendar not found: {args.calendar_name}"
        else:
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return "No default calendar found."

        # Get events
        start = datetime.now(UTC)
        end = start + timedelta(days=args.days_ahead)

        events = await self._client.get_events(calendar.caluid, start, end)
        if not events:
            return f"No events found in '{calendar.name}' for the next {args.days_ahead} days."

        lines = [f"Found {len(events)} event(s) in '{calendar.name}':\n"]
        for evt in events:
            start_str = evt.start.strftime("%Y-%m-%d %H:%M") if evt.start else "Unknown"
            end_str = evt.end.strftime("%H:%M") if evt.end else ""
            time_str = f"{start_str} - {end_str}" if end_str else start_str

            lines.append(f"- **{evt.title}**")
            lines.append(f"  Time: {time_str}")
            if evt.location:
                lines.append(f"  Location: {evt.location}")
            if evt.attendees:
                lines.append(f"  Attendees: {', '.join(evt.attendees)}")
            lines.append("")

        return "\n".join(lines)


class CheckAvailabilityTool(Tool):
    """Check calendar availability for a time range."""

    name = "check_availability"
    description = (
        "Check if a time slot is available on the calendar. Returns busy times "
        "within the specified range. Use this before scheduling meetings to "
        "ensure there are no conflicts."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date/time in ISO format (e.g., '2024-12-15T10:00:00')",
            },
            "end_date": {
                "type": "string",
                "description": "End date/time in ISO format (e.g., '2024-12-15T11:00:00')",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of attendee emails to check availability for",
            },
        },
        "required": ["start_date", "end_date"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Checking availability"

    def __init__(self, calendar_client: ZohoCalendarClient) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> str:
        """Check availability for a time range."""
        args = CheckAvailabilityArgs(**kwargs)

        try:
            start = datetime.fromisoformat(args.start_date.replace("Z", "+00:00"))
            end = datetime.fromisoformat(args.end_date.replace("Z", "+00:00"))
        except ValueError as e:
            return f"Invalid date format: {e}"

        # The freebusy API requires at least one email address
        if not args.attendees:
            # Instead of failing, check the calendar directly for conflicts
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return "No default calendar found to check availability."

            events = await self._client.get_events(calendar.caluid, start, end)
            if not events:
                time_range = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
                return f"The time slot {time_range} appears to be **available** (no events found)."

            # Check for overlapping events
            conflicts = []
            for evt in events:
                # Check if event overlaps with requested time
                if evt.start < end and evt.end > start:
                    conflicts.append(evt)

            if not conflicts:
                time_range = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
                return f"The time slot {time_range} is **available**."

            lines = ["The requested time has conflicts:\n"]
            for evt in conflicts:
                evt_time = (
                    f"{evt.start.strftime('%H:%M')}-{evt.end.strftime('%H:%M')}"
                    if evt.start and evt.end else "unknown time"
                )
                lines.append(f"- **{evt.title}** ({evt_time})")
            return "\n".join(lines)

        busy_slots = await self._client.check_availability(start, end, args.attendees)

        if not busy_slots:
            time_range = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
            return f"The time slot {time_range} is **available**."

        lines = ["The requested time has conflicts:\n"]
        for slot in busy_slots:
            slot_start = slot.get("start", "")
            slot_end = slot.get("end", "")
            lines.append(f"- Busy: {slot_start} to {slot_end}")

        return "\n".join(lines)


class CreateEventTool(Tool):
    """Create a new calendar event."""

    name = "create_event"
    description = (
        "Create a new event on a calendar. Specify the title, start/end times, "
        "and optionally a description, location, and attendees. "
        "Use check_availability first to ensure there are no conflicts."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Event title",
            },
            "start": {
                "type": "string",
                "description": "Start date/time in ISO format (e.g., '2024-12-15T10:00:00')",
            },
            "end": {
                "type": "string",
                "description": "End date/time in ISO format (e.g., '2024-12-15T11:00:00')",
            },
            "calendar_name": {
                "type": "string",
                "description": (
                    "Calendar name to create event on. Uses 'Default' if not specified. "
                    "Examples: 'Default', 'Studio A'"
                ),
            },
            "description": {
                "type": "string",
                "description": "Optional event description",
            },
            "location": {
                "type": "string",
                "description": "Optional event location",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of attendee email addresses",
            },
            "is_allday": {
                "type": "boolean",
                "description": "Whether this is an all-day event (default: false)",
            },
        },
        "required": ["title", "start", "end"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating calendar event"

    def __init__(self, calendar_client: ZohoCalendarClient) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> str:
        """Create a calendar event."""
        args = CreateEventArgs(**kwargs)

        # Find the calendar
        if args.calendar_name:
            calendar = await self._client.get_calendar_by_name(args.calendar_name)
            if not calendar:
                return f"Calendar not found: {args.calendar_name}"
        else:
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return "No default calendar found."

        # Parse dates
        try:
            start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        except ValueError as e:
            return f"Invalid date format: {e}"

        # Create the event
        event = await self._client.create_event(
            caluid=calendar.caluid,
            title=args.title,
            start=start,
            end=end,
            description=args.description,
            location=args.location,
            attendees=args.attendees,
            is_allday=args.is_allday,
        )

        if event:
            time_str = (
                f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
            )
            result = [
                f"Event created successfully on '{calendar.name}':\n",
                f"**{event.title}**",
                f"Time: {time_str}",
            ]
            if event.location:
                result.append(f"Location: {event.location}")
            if event.attendees:
                result.append(f"Attendees: {', '.join(event.attendees)}")
            return "\n".join(result)

        return "Failed to create event."


class FindFreeSlotsTool(Tool):
    """Find available time slots for meetings."""

    name = "find_free_slots"
    description = (
        "Find available time slots of a specified duration within the next N days. "
        "Use this to suggest meeting times when scheduling appointments. "
        "Returns a list of free time slots that can accommodate the requested duration."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "duration_minutes": {
                "type": "integer",
                "description": "Required slot duration in minutes (e.g., 30, 60)",
            },
            "days_ahead": {
                "type": "integer",
                "description": "Number of days ahead to search (default: 14)",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional attendee emails to consider for availability",
            },
        },
        "required": ["duration_minutes"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Finding free time slots"

    def __init__(self, calendar_client: ZohoCalendarClient) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> str:
        """Find free time slots."""
        args = FindFreeSlotsArgs(**kwargs)

        start = datetime.now(UTC)
        end = start + timedelta(days=args.days_ahead)

        free_slots = await self._client.find_free_slots(
            duration_minutes=args.duration_minutes,
            start=start,
            end=end,
            attendees=args.attendees,
        )

        if not free_slots:
            return (
                f"No free slots of {args.duration_minutes} minutes found "
                f"in the next {args.days_ahead} days."
            )

        lines = [
            f"Found {len(free_slots)} available slot(s) of {args.duration_minutes} minutes:\n"
        ]
        for slot in free_slots[:10]:  # Limit to first 10 slots
            slot_start = slot["start"]
            slot_end = slot["end"]
            lines.append(
                f"- {slot_start.strftime('%Y-%m-%d %H:%M')} to "
                f"{slot_end.strftime('%H:%M')}"
            )

        if len(free_slots) > 10:
            lines.append(f"\n... and {len(free_slots) - 10} more slots available.")

        return "\n".join(lines)


class UpdateEventTool(Tool):
    """Update an existing calendar event."""

    name = "update_event"
    description = (
        "Update an existing calendar event. Can change the title, time, description, "
        "or location. For recurring events, you can update all occurrences, just this "
        "one, or this and all future occurrences. First searches for the event by title, "
        "then updates it with the new values."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_title": {
                "type": "string",
                "description": "Title of the event to update (used to find the event)",
            },
            "calendar_name": {
                "type": "string",
                "description": "Calendar name where the event exists (optional)",
            },
            "new_title": {
                "type": "string",
                "description": "New title for the event (optional)",
            },
            "new_start": {
                "type": "string",
                "description": "New start date/time in ISO format, e.g., '2026-04-17T15:00:00'",
            },
            "new_end": {
                "type": "string",
                "description": "New end date/time in ISO format, e.g., '2026-04-17T16:00:00'",
            },
            "new_description": {
                "type": "string",
                "description": "New description for the event (optional)",
            },
            "new_location": {
                "type": "string",
                "description": "New location for the event (optional)",
            },
            "recurrence_edittype": {
                "type": "string",
                "enum": ["all", "following", "only"],
                "description": (
                    "For recurring events: 'all' updates all occurrences, "
                    "'following' updates this and future occurrences, "
                    "'only' updates just this occurrence. Default: 'all'"
                ),
            },
        },
        "required": ["event_title"],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return f"Updating event: {arguments.get('event_title', 'unknown')}"

    def __init__(self, calendar_client: ZohoCalendarClient) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> str:
        """Update a calendar event."""
        args = UpdateEventArgs(**kwargs)

        # Find the calendar
        if args.calendar_name:
            calendar = await self._client.get_calendar_by_name(args.calendar_name)
            if not calendar:
                return f"Calendar not found: {args.calendar_name}"
        else:
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return "No default calendar found."

        # Search for the event by title (Zoho API limits range to 31 days)
        start = datetime.now(UTC)
        end = start + timedelta(days=30)

        events = await self._client.get_events(calendar.caluid, start, end)
        matching_events = [
            e for e in events
            if args.event_title.lower() in e.title.lower()
        ]

        if not matching_events:
            return (
                f"No event found matching '{args.event_title}' in calendar '{calendar.name}'. "
                "Please check the event title and try again."
            )

        # Use the first matching event - this is the occurrence in the search range
        event = matching_events[0]
        logger.info(
            "Found event from search: uid=%s, title=%s, start=%s",
            event.uid, event.title, event.start
        )

        # Get full event details including etag
        full_event = await self._client.get_event(calendar.caluid, event.uid)
        if not full_event or not full_event.etag:
            return f"Could not retrieve event details for '{event.title}'."

        logger.info(
            "Full event details: uid=%s, start=%s, recurrenceid=%s, is_recurring=%s",
            full_event.uid, full_event.start, full_event.recurrenceid, full_event.is_recurring
        )

        # Parse new dates if provided, otherwise use existing event times
        # dateandtime is required for all updates (especially recurring events)
        if args.new_start:
            try:
                new_start = datetime.fromisoformat(args.new_start.replace("Z", "+00:00"))
            except ValueError as e:
                return f"Invalid start date format: {e}"
        else:
            new_start = full_event.start
            if not new_start:
                return f"Could not determine start time for event '{event.title}'."

        if args.new_end:
            try:
                new_end = datetime.fromisoformat(args.new_end.replace("Z", "+00:00"))
            except ValueError as e:
                return f"Invalid end date format: {e}"
        else:
            new_end = full_event.end
            if not new_end:
                return f"Could not determine end time for event '{event.title}'."

        # For recurring events:
        # - recurrenceid must be the start time of the TARGET occurrence being updated
        # - "all" edittype: Updates all occurrences (no recurrenceid needed)
        # - "following"/"only": Updates this and future / only this occurrence
        effective_edittype = args.recurrence_edittype

        # For recurring events, if user wants to change time with edittype "all",
        # switch to "following" since changing dateandtime for "all" causes API errors
        if (
            full_event.is_recurring
            and effective_edittype == "all"
            and (args.new_start or args.new_end)
        ):
            logger.info(
                "Switching from 'all' to 'following' for recurring event time change"
            )
            effective_edittype = "following"

        # For "following" or "only" on recurring events, generate recurrenceid from
        # the TARGET occurrence's start time from the SEARCH result (event.start),
        # not from get_event which returns the master event's start date
        recurrenceid = None
        if full_event.is_recurring and effective_edittype in ("following", "only"):
            # Use the occurrence start from the search result, not the master event
            occurrence_start = event.start
            if occurrence_start:
                start_utc = occurrence_start.astimezone(UTC)
                recurrenceid = start_utc.strftime("%Y%m%dT%H%M%SZ")
                logger.info(
                    "Generated recurrenceid from search occurrence: %s (from %s)",
                    recurrenceid, occurrence_start
                )

        if full_event.is_recurring:
            logger.info(
                "Recurring event detected: recurrenceid=%s, edittype=%s",
                recurrenceid, effective_edittype
            )

        # Update the event
        updated_event = await self._client.update_event(
            caluid=calendar.caluid,
            event_uid=event.uid,
            etag=full_event.etag,
            title=args.new_title,
            start=new_start,
            end=new_end,
            tz=full_event.timezone,
            description=args.new_description,
            location=args.new_location,
            is_allday=full_event.is_allday,
            recurrence_edittype=effective_edittype,
            recurrenceid=recurrenceid,
            rrule=full_event.rrule,
            is_recurring=full_event.is_recurring,
        )

        if updated_event:
            changes = []
            if args.new_title:
                changes.append(f"Title: {args.new_title}")
            if new_start and new_end:
                changes.append(
                    f"Time: {new_start.strftime('%Y-%m-%d %H:%M')} to {new_end.strftime('%H:%M')}"
                )
            if args.new_location:
                changes.append(f"Location: {args.new_location}")
            if args.new_description:
                changes.append("Description updated")

            edit_scope = {
                "all": "all occurrences",
                "following": "this and future occurrences",
                "only": "only this occurrence",
            }.get(args.recurrence_edittype, "all occurrences")

            result = [
                f"Event updated successfully ({edit_scope}):\n",
                f"**{updated_event.title}**",
            ]
            if changes:
                result.append("Changes:")
                result.extend([f"  - {c}" for c in changes])
            return "\n".join(result)

        return "Failed to update event."
