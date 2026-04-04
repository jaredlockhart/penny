"""Zoho plugin for Penny.

Provides email, calendar, and project management via Zoho APIs.

Required environment variables:
    ZOHO_API_ID       — Zoho OAuth client ID
    ZOHO_API_SECRET   — Zoho OAuth client secret
    ZOHO_REFRESH_TOKEN — Zoho OAuth refresh token
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from penny.plugins import CAPABILITY_CALENDAR, CAPABILITY_EMAIL, CAPABILITY_PROJECT, Plugin
from penny.plugins.zoho.commands import (
    ZohoCalendarCommand,
    ZohoEmailCommand,
    ZohoProjectCommand,
)

if TYPE_CHECKING:
    from penny.commands.base import Command
    from penny.config import Config
    from penny.tools.base import Tool


class ZohoPlugin(Plugin):
    """Zoho integration plugin for email, calendar, and projects."""

    name = "zoho"
    capabilities = [CAPABILITY_EMAIL, CAPABILITY_CALENDAR, CAPABILITY_PROJECT]

    def __init__(self, config: Config) -> None:
        self._client_id = os.environ["ZOHO_API_ID"]
        self._client_secret = os.environ["ZOHO_API_SECRET"]
        self._refresh_token = os.environ["ZOHO_REFRESH_TOKEN"]

    @classmethod
    def is_configured(cls, config: Config) -> bool:
        """Return True if all Zoho credentials are present."""
        return bool(
            os.getenv("ZOHO_API_ID")
            and os.getenv("ZOHO_API_SECRET")
            and os.getenv("ZOHO_REFRESH_TOKEN")
        )

    def get_commands(self) -> list[Command]:
        """Return Zoho commands for email, calendar, and projects."""
        return [
            ZohoEmailCommand(
                client_id=self._client_id,
                client_secret=self._client_secret,
                refresh_token=self._refresh_token,
            ),
            ZohoCalendarCommand(
                client_id=self._client_id,
                client_secret=self._client_secret,
                refresh_token=self._refresh_token,
            ),
            ZohoProjectCommand(
                client_id=self._client_id,
                client_secret=self._client_secret,
                refresh_token=self._refresh_token,
            ),
        ]

    def get_tools(self) -> list[Tool]:
        """Zoho tools are created per-request inside command handlers."""
        return []


PLUGIN_CLASS = ZohoPlugin
