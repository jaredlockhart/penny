"""InvoiceNinja commands — slash commands contributed by the InvoiceNinja plugin."""

from __future__ import annotations

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult

logger = logging.getLogger(__name__)


class InvoiceCommand(Command):
    """Query InvoiceNinja invoices and financial data."""

    name = "invoice"
    description = "Search and query your InvoiceNinja invoices"
    help_text = (
        "Usage: /invoice <question>\n\n"
        "Ask questions about your invoices and Penny will find the answer.\n\n"
        "Examples:\n"
        "• /invoice which invoices are overdue\n"
        "• /invoice total revenue this month\n"
        "• /invoice list unpaid invoices"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the invoice command."""
        raise NotImplementedError("InvoiceNinja plugin is not yet fully implemented")
