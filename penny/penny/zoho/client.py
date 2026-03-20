"""Zoho Mail API client."""

from __future__ import annotations

import html.parser
import logging
import time
from typing import Any

import httpx

from penny.constants import PennyConstants
from penny.jmap.models import EmailAddress, EmailDetail, EmailSummary
from penny.zoho.models import ZohoAccount, ZohoFolder, ZohoSession

logger = logging.getLogger(__name__)

EMAIL_SEARCH_LIMIT = 10


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Simple HTML tag stripper."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html_text: str) -> str:
    """Strip HTML tags and return plain text."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html_text)
    return extractor.get_text()


class ZohoClient:
    """Zoho Mail API client.

    Uses OAuth 2.0 with client credentials to access Zoho Mail API.
    Requires a refresh token to be obtained via the OAuth flow.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        timeout: float,
        max_body_length: int,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._max_body_length = max_body_length
        self._session: ZohoSession | None = None
        self._account: ZohoAccount | None = None
        self._folders: list[ZohoFolder] | None = None
        self._http = httpx.AsyncClient(timeout=timeout)

    async def _ensure_access_token(self) -> str:
        """Ensure we have a valid access token, refreshing if needed."""
        now = time.time()
        if self._session and self._session.expires_at > now + 60:
            return self._session.access_token

        resp = await self._http.post(
            PennyConstants.ZOHO_TOKEN_URL,
            data={
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Zoho OAuth error: {data.get('error')}")

        expires_in = data.get("expires_in", 3600)
        self._session = ZohoSession(
            access_token=data["access_token"],
            expires_at=now + expires_in,
        )
        logger.info("Zoho access token refreshed, expires in %ds", expires_in)
        return self._session.access_token

    async def _get_headers(self) -> dict[str, str]:
        """Get headers with current access token."""
        token = await self._ensure_access_token()
        return {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _ensure_account(self) -> ZohoAccount:
        """Fetch and cache the primary Zoho Mail account."""
        if self._account:
            return self._account

        headers = await self._get_headers()
        resp = await self._http.get(PennyConstants.ZOHO_ACCOUNTS_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        accounts = data.get("data", [])
        if not accounts:
            raise RuntimeError("No Zoho Mail accounts found")

        # Use the first (primary) account
        acct = accounts[0]

        # emailAddress can be a list of dicts or a single dict
        email_addr_field = acct.get("emailAddress", [])
        if isinstance(email_addr_field, list) and email_addr_field:
            email_address = email_addr_field[0].get("mailId", "")
        elif isinstance(email_addr_field, dict):
            email_address = email_addr_field.get("mailId", "")
        else:
            email_address = ""

        self._account = ZohoAccount(
            account_id=str(acct["accountId"]),
            email_address=email_address,
            display_name=acct.get("displayName"),
        )
        logger.info(
            "Zoho account: %s (%s)",
            self._account.email_address,
            self._account.account_id,
        )
        return self._account

    async def get_folders(self) -> list[ZohoFolder]:
        """Fetch and cache all folders for the account."""
        if self._folders is not None:
            return self._folders

        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/folders"
        resp = await self._http.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        folders_data = data.get("data", [])
        self._folders = [
            ZohoFolder(
                folder_id=str(f["folderId"]),
                folder_name=f.get("folderName", ""),
                folder_type=f.get("folderType", ""),
                path=f.get("path", ""),
                is_archived=bool(f.get("isArchived", 0)),
            )
            for f in folders_data
        ]
        logger.info("Loaded %d Zoho folders", len(self._folders))
        return self._folders

    async def get_folder_by_name(self, name: str) -> ZohoFolder | None:
        """Get a folder by name (case-insensitive)."""
        folders = await self.get_folders()
        name_lower = name.lower()
        for folder in folders:
            if folder.folder_name.lower() == name_lower:
                return folder
        return None

    async def get_folder_by_type(self, folder_type: str) -> ZohoFolder | None:
        """Get a folder by type (e.g., 'Inbox', 'Sent', 'Drafts')."""
        folders = await self.get_folders()
        for folder in folders:
            if folder.folder_type == folder_type:
                return folder
        return None

    async def list_emails(
        self,
        folder_name: str | None = None,
        limit: int = EMAIL_SEARCH_LIMIT,
    ) -> list[EmailSummary]:
        """List emails from a specific folder.

        Args:
            folder_name: Name of folder to list (default: Inbox)
            limit: Maximum number of emails to return
        """
        account = await self._ensure_account()
        headers = await self._get_headers()

        # Get the folder ID
        if folder_name:
            folder = await self.get_folder_by_name(folder_name)
            if not folder:
                logger.warning("Folder not found: %s", folder_name)
                return []
            folder_id = folder.folder_id
        else:
            # Default to Inbox
            folder = await self.get_folder_by_type("Inbox")
            if not folder:
                logger.warning("Inbox folder not found")
                return []
            folder_id = folder.folder_id

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages/view"
        params = {
            "folderId": folder_id,
            "limit": limit,
            "includeto": "true",
        }

        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        emails_data = data.get("data", [])
        logger.info("Listed %d email(s) from folder %s", len(emails_data), folder_name or "Inbox")

        return [self._parse_email_summary(e) for e in emails_data]

    async def search_emails(
        self,
        text: str | None = None,
        from_addr: str | None = None,
        subject: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = EMAIL_SEARCH_LIMIT,
    ) -> list[EmailSummary]:
        """Search emails and return summaries.

        Uses Zoho's search syntax to build the searchKey parameter.
        Zoho syntax: parameter:value with :: between multiple conditions.
        Docs: https://www.zoho.com/mail/help/search-syntax.html
        """
        search_parts = []
        if text:
            # Use 'entire:' for full-text search across all email content
            # Quote if contains spaces for exact phrase matching
            if " " in text:
                search_parts.append(f'entire:"{text}"')
            else:
                search_parts.append(f"entire:{text}")
        if from_addr:
            search_parts.append(f"sender:{from_addr}")
        if subject:
            # Quote subjects containing spaces or special characters
            if " " in subject or ":" in subject:
                search_parts.append(f'subject:"{subject}"')
            else:
                search_parts.append(f"subject:{subject}")
        # Note: Zoho date format is DD-MMM-YYYY (e.g., 12-Sep-2017)
        # Skip date filters if format doesn't match to avoid empty results
        if after and self._is_valid_zoho_date(after):
            search_parts.append(f"fromDate:{after}")
        if before and self._is_valid_zoho_date(before):
            search_parts.append(f"toDate:{before}")

        # Join with :: for AND logic between conditions
        search_key = "::".join(search_parts) if search_parts else "newMails"
        logger.info("Zoho search query: %s", search_key)

        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages/search"
        params = {
            "searchKey": search_key,
            "limit": limit,
            "includeto": "true",
        }

        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        emails_data = data.get("data", [])
        logger.info("Zoho search returned %d email(s)", len(emails_data))

        return [self._parse_email_summary(e) for e in emails_data]

    def _parse_email_summary(self, e: dict[str, Any]) -> EmailSummary:
        """Parse a Zoho email response into an EmailSummary."""
        from_addr = e.get("fromAddress", "")
        from_name = e.get("sender", "")

        return EmailSummary(
            id=self._make_email_id(e),
            subject=e.get("subject", "(no subject)"),
            from_addresses=[EmailAddress(name=from_name or None, email=from_addr)],
            received_at=self._format_timestamp(e.get("receivedTime", 0)),
            preview=e.get("summary", ""),
        )

    def _make_email_id(self, e: dict[str, Any]) -> str:
        """Create an ID from the message URI or folderId:messageId."""
        # Prefer URI as it's the canonical way to fetch the message
        uri = e.get("URI", "")
        if uri:
            return uri
        # Fallback to folder:message format
        folder_id = e.get("folderId", "")
        message_id = e.get("messageId", "")
        return f"{folder_id}:{message_id}"

    def _parse_email_id(self, email_id: str) -> tuple[str | None, str]:
        """Parse an email ID into (uri_or_none, folder:message_or_id).

        If email_id is a URI, returns (uri, "").
        If email_id is folder:message format, returns (None, email_id).
        """
        if email_id.startswith("http"):
            return email_id, ""
        if ":" in email_id:
            return None, email_id
        return None, email_id

    def _format_timestamp(self, ts: int | str) -> str:
        """Format a Unix timestamp (ms) to ISO 8601."""
        if not ts:
            return ""
        try:
            ts_int = int(ts)
            from datetime import UTC, datetime

            dt = datetime.fromtimestamp(ts_int / 1000, tz=UTC)
            return dt.isoformat()
        except ValueError, TypeError:
            return str(ts)

    @staticmethod
    def _is_valid_zoho_date(date_str: str) -> bool:
        """Check if date string is in Zoho format DD-MMM-YYYY (e.g., 12-Sep-2017)."""
        import re

        # Zoho expects DD-MMM-YYYY format
        pattern = r"^\d{1,2}-[A-Za-z]{3}-\d{4}$"
        return bool(re.match(pattern, date_str))

    async def read_emails(self, email_ids: list[str]) -> list[EmailDetail]:
        """Fetch full email bodies by IDs."""
        if not email_ids:
            return []

        headers = await self._get_headers()
        results: list[EmailDetail] = []

        for email_id in email_ids:
            try:
                detail = await self._fetch_email_detail(email_id, headers)
                if detail:
                    results.append(detail)
            except Exception as e:
                logger.warning("Failed to fetch email %s: %s", email_id, e)

        return results

    async def _fetch_email_detail(
        self,
        email_id: str,
        headers: dict[str, str],
    ) -> EmailDetail | None:
        """Fetch a single email's full content using the content endpoint."""
        # Extract folder_id and message_id from the composite ID
        if ":" not in email_id:
            logger.warning("Invalid email ID format (no colon): %s", email_id)
            return None

        folder_id, message_id = email_id.split(":", 1)
        if not folder_id or not message_id:
            logger.warning("Invalid email ID format (empty parts): %s", email_id)
            return None

        account = await self._ensure_account()

        # Use the content endpoint with folderId from search results
        content_url = (
            f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}"
            f"/folders/{folder_id}/messages/{message_id}/content"
        )

        logger.debug("Fetching email content from: %s", content_url)
        resp = await self._http.get(
            content_url, headers=headers, params={"includeBlockContent": "true"}
        )
        resp.raise_for_status()
        content_data = resp.json().get("data", {})

        text_body = content_data.get("content", "")

        # Strip HTML if content appears to be HTML
        if text_body and "<" in text_body:
            text_body = _strip_html(text_body)

        # Truncate long bodies
        if len(text_body) > self._max_body_length:
            text_body = text_body[: self._max_body_length] + "\n\n[truncated]"

        # Get metadata from content response or use defaults
        from_addr = content_data.get("fromAddress", "")
        from_name = content_data.get("sender", "")
        to_list = (
            content_data.get("toAddress", "").split(",") if content_data.get("toAddress") else []
        )
        subject = content_data.get("subject", "(no subject)")
        received_time = content_data.get("receivedTime", 0)

        return EmailDetail(
            id=email_id,
            subject=subject,
            from_addresses=[EmailAddress(name=from_name or None, email=from_addr)],
            to_addresses=[EmailAddress(email=addr.strip()) for addr in to_list if addr.strip()],
            received_at=self._format_timestamp(received_time),
            text_body=text_body,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
