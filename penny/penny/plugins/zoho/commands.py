"""Zoho commands — email, calendar, and project management."""

from __future__ import annotations

import logging

from penny.agents.base import Agent
from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.plugins.zoho.calendar_client import ZohoCalendarClient
from penny.plugins.zoho.calendar_tools import (
    CheckAvailabilityTool,
    CreateEventTool,
    FindFreeSlotsTool,
    GetEventsTool,
    ListCalendarsTool,
    UpdateEventTool,
)
from penny.plugins.zoho.client import ZohoClient
from penny.plugins.zoho.project_tools import (
    CreateProjectTool,
    CreateTaskListTool,
    CreateTaskTool,
    GetProjectDetailsTool,
    ListProjectsTool,
    ListTaskListsTool,
    ListTasksTool,
    UpdateTaskTool,
)
from penny.plugins.zoho.projects_client import ZohoProjectsClient
from penny.plugins.zoho.tools import (
    ApplyLabelTool,
    CreateFolderTool,
    DraftEmailTool,
    ListEmailsTool,
    ListFoldersTool,
    ListLabelsTool,
    MoveEmailsTool,
)
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

logger = logging.getLogger(__name__)


class ZohoEmailCommand(Command):
    """Search Zoho email and answer questions about it.

    This command is registered as a provider under the /email routing layer.
    It can also be accessed directly as /zoho for backward compatibility.
    """

    name = "zoho"
    description = "Search your Zoho email and answer questions"
    help_text = (
        "Usage: /email zoho <question>  (or /zoho <question>)\n\n"
        "Ask a question about your Zoho email and Penny will search and read "
        "relevant messages to find the answer.\n\n"
        "Examples:\n"
        "• /email zoho what packages am I expecting\n"
        "• /email zoho when is my dentist appointment\n"
        "• /email zoho any emails from mom this week"
    )

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the Zoho email command."""
        prompt = args.strip()
        if not prompt:
            return CommandResult(text=PennyResponse.EMAIL_NO_QUERY_TEXT)

        zoho_client = ZohoClient(
            self._client_id,
            self._client_secret,
            self._refresh_token,
            timeout=context.config.runtime.JMAP_REQUEST_TIMEOUT,
            max_body_length=int(context.config.runtime.EMAIL_BODY_MAX_LENGTH),
        )
        agent: Agent | None = None
        try:
            tools: list[Tool] = [
                SearchEmailsTool(zoho_client),
                ListEmailsTool(zoho_client),
                ListFoldersTool(zoho_client),
                ReadEmailsTool(zoho_client),
                DraftEmailTool(zoho_client),
                MoveEmailsTool(zoho_client),
                CreateFolderTool(zoho_client),
                ApplyLabelTool(zoho_client),
                ListLabelsTool(zoho_client),
            ]
            agent = Agent(
                system_prompt=Prompt.ZOHO_SYSTEM_PROMPT,
                model_client=context.model_client,
                tools=tools,
                db=context.db,
                config=context.config,
                allow_repeat_tools=True,
            )
            response = await agent.run(prompt, max_steps=context.config.email_max_steps)
            return CommandResult(text=response.answer)
        except Exception as e:
            logger.exception("Zoho email search failed")
            return CommandResult(text=PennyResponse.EMAIL_ERROR.format(error=e))
        finally:
            if agent:
                await agent.close()
            await zoho_client.close()


class ZohoCalendarCommand(Command):
    """Manage Zoho Calendar — check availability, create events, find free slots.

    This command provides calendar management capabilities including:
    - Listing calendars
    - Viewing upcoming events
    - Checking availability for time slots
    - Creating new events
    - Finding free time slots for meetings
    """

    name = "calendar"
    description = "Manage your Zoho Calendar — check availability and create events"
    help_text = (
        "Usage: /calendar <request>\n\n"
        "Manage your Zoho Calendar. Penny can check availability, create events, "
        "and find free time slots for meetings.\n\n"
        "Examples:\n"
        "• /calendar what's on my schedule this week\n"
        "• /calendar am I free on Friday at 2pm\n"
        "• /calendar create a meeting with John on Monday at 10am\n"
        "• /calendar find a 1 hour slot for a meeting next week\n"
        "• /calendar create session for Band X in Studio A on December 2nd"
    )

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the calendar command."""
        prompt = args.strip()
        if not prompt:
            return CommandResult(
                text="Please specify what you'd like to do with your calendar. "
                "For example: 'what's on my schedule this week' or "
                "'am I free on Friday at 2pm'"
            )

        calendar_client = ZohoCalendarClient(
            self._client_id,
            self._client_secret,
            self._refresh_token,
        )
        agent: Agent | None = None
        try:
            tools: list[Tool] = [
                ListCalendarsTool(calendar_client),
                GetEventsTool(calendar_client),
                CheckAvailabilityTool(calendar_client),
                CreateEventTool(calendar_client),
                UpdateEventTool(calendar_client),
                FindFreeSlotsTool(calendar_client),
            ]
            agent = Agent(
                system_prompt=Prompt.ZOHO_CALENDAR_SYSTEM_PROMPT,
                model_client=context.model_client,
                tools=tools,
                db=context.db,
                config=context.config,
                allow_repeat_tools=True,
            )
            response = await agent.run(prompt, max_steps=context.config.email_max_steps)
            return CommandResult(text=response.answer)
        except Exception as e:
            logger.exception("Zoho calendar command failed")
            return CommandResult(text=f"Calendar error: {e}")
        finally:
            if agent:
                await agent.close()
            await calendar_client.close()


class ZohoProjectCommand(Command):
    """Manage Zoho Projects — create projects, tasks, and track progress.

    This command provides project management capabilities including:
    - Listing and creating projects
    - Managing task lists (milestones)
    - Creating and updating tasks
    - Assigning tasks to team members
    - Tracking task progress
    """

    name = "project"
    description = "Manage Zoho Projects — create projects, tasks, and track progress"
    help_text = (
        "Usage: /project <request>\n\n"
        "Manage your Zoho Projects. Penny can create projects, manage tasks, "
        "and help you stay organized.\n\n"
        "Examples:\n"
        "• /project list all projects\n"
        "• /project create a new project called 'Website Redesign'\n"
        "• /project add a task 'Design homepage' to Website Redesign\n"
        "• /project what tasks are in the Website Redesign project\n"
        "• /project mark 'Design homepage' as 50% complete\n"
        "• /project assign 'Design homepage' to John"
    )

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the project command."""
        prompt = args.strip()
        if not prompt:
            return CommandResult(
                text="Please specify what you'd like to do with your projects. "
                "For example: 'list all projects' or "
                "'create a task in Website Redesign'"
            )

        projects_client = ZohoProjectsClient(
            self._client_id,
            self._client_secret,
            self._refresh_token,
        )
        agent: Agent | None = None
        try:
            tools: list[Tool] = [
                ListProjectsTool(projects_client),
                GetProjectDetailsTool(projects_client),
                CreateProjectTool(projects_client),
                ListTaskListsTool(projects_client),
                CreateTaskListTool(projects_client),
                ListTasksTool(projects_client),
                CreateTaskTool(projects_client),
                UpdateTaskTool(projects_client),
            ]
            agent = Agent(
                system_prompt=Prompt.ZOHO_PROJECT_SYSTEM_PROMPT,
                model_client=context.model_client,
                tools=tools,
                db=context.db,
                config=context.config,
                allow_repeat_tools=True,
            )
            response = await agent.run(prompt, max_steps=context.config.email_max_steps)
            return CommandResult(text=response.answer)
        except Exception as e:
            logger.exception("Zoho project command failed")
            return CommandResult(text=f"Project error: {e}")
        finally:
            if agent:
                await agent.close()
            await projects_client.close()
