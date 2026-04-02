"""Zoho Mail plugin for Penny.

Provides email search, listing, reading, and drafting via the Zoho Mail API.

Required environment variables:
    ZOHO_API_ID       — Zoho OAuth client ID
    ZOHO_API_SECRET   — Zoho OAuth client secret
    ZOHO_REFRESH_TOKEN — Zoho OAuth refresh token
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from penny.plugins import CAPABILITY_EMAIL, Plugin
from penny.plugins.zoho.commands import ZohoEmailCommand

if TYPE_CHECKING:
    from penny.commands.base import Command
    from penny.config import Config
    from penny.tools.base import Tool


class ZohoPlugin(Plugin):
    """Zoho Mail integration plugin."""

    name = "zoho"
    capabilities = [CAPABILITY_EMAIL]

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
        """Return the Zoho email command."""
        return [
            ZohoEmailCommand(
                client_id=self._client_id,
                client_secret=self._client_secret,
                refresh_token=self._refresh_token,
            )
        ]

    def get_tools(self) -> list[Tool]:
        """Zoho tools are created per-request inside ZohoEmailCommand."""
        return []


PLUGIN_CLASS = ZohoPlugin
