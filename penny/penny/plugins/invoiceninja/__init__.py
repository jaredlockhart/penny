"""InvoiceNinja plugin for Penny.

Provides invoice querying and financial reporting via InvoiceNinja v5 API.

Required environment variables:
    INVOICENINJA_API_TOKEN — InvoiceNinja API token
    INVOICENINJA_URL       — InvoiceNinja instance URL (e.g. https://invoicing.co)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from penny.plugins import CAPABILITY_INVOICING, Plugin
from penny.plugins.invoiceninja.commands import InvoiceCommand
from penny.plugins.invoiceninja.tools import ListInvoicesTool

if TYPE_CHECKING:
    from penny.commands.base import Command
    from penny.config import Config
    from penny.tools.base import Tool


class InvoiceNinjaPlugin(Plugin):
    """InvoiceNinja integration plugin."""

    name = "invoiceninja"
    capabilities = [CAPABILITY_INVOICING]

    def __init__(self, config: Config) -> None:
        from penny.plugins.invoiceninja.client import InvoiceNinjaClient

        self._client = InvoiceNinjaClient(
            api_token=os.environ["INVOICENINJA_API_TOKEN"],
            base_url=os.environ["INVOICENINJA_URL"],
        )

    @classmethod
    def is_configured(cls, config: Config) -> bool:
        """Return True if InvoiceNinja credentials are present."""
        return bool(os.getenv("INVOICENINJA_API_TOKEN") and os.getenv("INVOICENINJA_URL"))

    def get_commands(self) -> list[Command]:
        """Return the invoice command."""
        return [InvoiceCommand()]

    def get_tools(self) -> list[Tool]:
        """Return LLM-callable InvoiceNinja tools."""
        return [ListInvoicesTool(self._client)]

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()


PLUGIN_CLASS = InvoiceNinjaPlugin
