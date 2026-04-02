"""Fastmail JMAP plugin for Penny.

Provides email search and reading via the Fastmail JMAP API.

Required environment variables:
    FASTMAIL_API_TOKEN — Fastmail API token (from Settings → Security → API Tokens)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from penny.plugins import CAPABILITY_EMAIL, Plugin
from penny.plugins.fastmail.commands import FastmailEmailCommand

if TYPE_CHECKING:
    from penny.commands.base import Command
    from penny.config import Config
    from penny.tools.base import Tool


class FastmailPlugin(Plugin):
    """Fastmail JMAP integration plugin."""

    name = "fastmail"
    capabilities = [CAPABILITY_EMAIL]

    def __init__(self, config: Config) -> None:
        self._api_token = os.environ["FASTMAIL_API_TOKEN"]

    @classmethod
    def is_configured(cls, config: Config) -> bool:
        """Return True if the Fastmail API token is present."""
        return bool(os.getenv("FASTMAIL_API_TOKEN"))

    def get_commands(self) -> list[Command]:
        """Return the Fastmail email command."""
        return [FastmailEmailCommand(api_token=self._api_token)]

    def get_tools(self) -> list[Tool]:
        """Fastmail tools are created per-request inside FastmailEmailCommand."""
        return []


PLUGIN_CLASS = FastmailPlugin
