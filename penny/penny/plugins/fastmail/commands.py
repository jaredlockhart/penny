"""Fastmail command — email provider implementation for the /email command."""

from __future__ import annotations

import logging

from penny.agents.base import Agent
from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.plugins.fastmail.client import JmapClient
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools import Tool
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

logger = logging.getLogger(__name__)


class FastmailEmailCommand(Command):
    """Search Fastmail email and answer questions about it.

    This command is registered as a provider under the /email routing layer.
    """

    name = "fastmail"
    description = "Search your Fastmail email and answer questions"
    help_text = (
        "Usage: /email fastmail <question>\n\n"
        "Ask a question about your Fastmail email and Penny will search and read "
        "relevant messages to find the answer.\n\n"
        "Examples:\n"
        "• /email fastmail what packages am I expecting\n"
        "• /email fastmail when is my dentist appointment\n"
        "• /email fastmail any emails from mom this week"
    )

    def __init__(self, api_token: str) -> None:
        self._api_token = api_token

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the Fastmail email command."""
        prompt = args.strip()
        if not prompt:
            return CommandResult(text=PennyResponse.EMAIL_NO_QUERY_TEXT)

        jmap_client = JmapClient(
            self._api_token,
            timeout=context.config.runtime.JMAP_REQUEST_TIMEOUT,
            max_body_length=int(context.config.runtime.EMAIL_BODY_MAX_LENGTH),
        )
        agent: Agent | None = None
        try:
            tools: list[Tool] = [
                SearchEmailsTool(jmap_client),
                ReadEmailsTool(jmap_client),
            ]
            agent = Agent(
                system_prompt=Prompt.EMAIL_SYSTEM_PROMPT,
                model_client=context.model_client,
                tools=tools,
                db=context.db,
                config=context.config,
                tool_timeout=context.config.tool_timeout,
                allow_repeat_tools=True,
            )
            response = await agent.run(prompt, max_steps=context.config.email_max_steps)
            return CommandResult(text=response.answer)
        except Exception as e:
            logger.exception("Fastmail email search failed")
            return CommandResult(text=PennyResponse.EMAIL_ERROR.format(error=e))
        finally:
            if agent:
                await agent.close()
            await jmap_client.close()
