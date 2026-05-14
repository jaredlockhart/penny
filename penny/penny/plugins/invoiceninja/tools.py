"""InvoiceNinja tools — LLM-callable tools for the InvoiceNinja plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.plugins.invoiceninja.client import InvoiceNinjaClient

logger = logging.getLogger(__name__)


class ListInvoicesTool(Tool):
    """List invoices from InvoiceNinja."""

    name = "list_invoices"
    description = (
        "List invoices from InvoiceNinja. Returns invoice numbers, client names, "
        "amounts, statuses, and due dates."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": (
                    "Filter by status: 'draft', 'sent', 'partial', 'paid', 'overdue'. "
                    "Omit to return all invoices."
                ),
            },
        },
        "required": [],
    }

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing invoices"

    def __init__(self, client: InvoiceNinjaClient) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError("InvoiceNinja plugin is not yet fully implemented")
