"""InvoiceNinja API client."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class InvoiceNinjaClient:
    """InvoiceNinja v5 API client.

    Requires an API token from Settings → API Tokens in InvoiceNinja.
    Set INVOICENINJA_API_TOKEN and INVOICENINJA_URL in your .env.
    """

    def __init__(self, api_token: str, base_url: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "X-Api-Token": api_token,
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
