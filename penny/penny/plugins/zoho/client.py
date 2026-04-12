"""Zoho Mail API client."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from penny.constants import PennyConstants
from penny.email.models import EmailAddress, EmailDetail, EmailSummary
from penny.html_utils import strip_html
from penny.plugins.zoho.models import ZohoAccount, ZohoFolder, ZohoSession

logger = logging.getLogger(__name__)

EMAIL_SEARCH_LIMIT = 10


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
        search_limit: int,
        list_limit: int,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._max_body_length = max_body_length
        self._search_limit = search_limit
        self._list_limit = list_limit
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

        acct = accounts[0]
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
    ) -> list[EmailSummary]:
        """List emails from a specific folder."""
        account = await self._ensure_account()
        headers = await self._get_headers()

        if folder_name:
            folder = await self.get_folder_by_name(folder_name)
            if not folder:
                logger.warning("Folder not found: %s", folder_name)
                return []
            folder_id = folder.folder_id
        else:
            folder = await self.get_folder_by_type("Inbox")
            if not folder:
                logger.warning("Inbox folder not found")
                return []
            folder_id = folder.folder_id

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages/view"
        params = {"folderId": folder_id, "limit": limit, "includeto": "true"}

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
    ) -> list[EmailSummary]:
        """Search emails and return summaries."""
        search_parts = []
        if text:
            search_parts.append(f'entire:"{text}"' if " " in text else f"entire:{text}")
        if from_addr:
            search_parts.append(f"sender:{from_addr}")
        if subject:
            needs_quotes = " " in subject or ":" in subject
            subject_part = f'subject:"{subject}"' if needs_quotes else f"subject:{subject}"
            search_parts.append(subject_part)
        if after and self._is_valid_zoho_date(after):
            search_parts.append(f"fromDate:{after}")
        if before and self._is_valid_zoho_date(before):
            search_parts.append(f"toDate:{before}")

        search_key = "::".join(search_parts) if search_parts else "newMails"
        logger.info("Zoho search query: %s", search_key)

        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages/search"
        params = {"searchKey": search_key, "limit": limit, "includeto": "true"}

        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        emails_data = data.get("data", [])
        logger.info("Zoho search returned %d email(s)", len(emails_data))

        if emails_data:
            logger.info("[DIAG] First email raw keys: %s", list(emails_data[0].keys()))

        summaries = [self._parse_email_summary(e) for e in emails_data]
        logger.info("[DIAG] Returning email IDs: %s", [s.id for s in summaries])
        return summaries

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
        uri = e.get("URI", "")
        if uri:
            return uri
        folder_id = e.get("folderId") or e.get("mailFolderId") or ""
        message_id = e.get("messageId", "")
        if not folder_id:
            logger.warning(
                "Missing folderId in email data. Available keys: %s, messageId: %s",
                list(e.keys()),
                message_id,
            )
        return f"{folder_id}:{message_id}"

    def _parse_email_id(self, email_id: str) -> tuple[str | None, str]:
        """Parse an email ID into (uri_or_none, folder:message_or_id)."""
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
            dt = datetime.fromtimestamp(ts_int / 1000, tz=UTC)
            return dt.isoformat()
        except ValueError, TypeError:
            return str(ts)

    @staticmethod
    def _is_valid_zoho_date(date_str: str) -> bool:
        """Check if date string is in Zoho format DD-MMM-YYYY (e.g., 12-Sep-2017)."""
        pattern = r"^\d{1,2}-[A-Za-z]{3}-\d{4}$"
        return bool(re.match(pattern, date_str))

    async def read_emails(self, email_ids: list[str]) -> list[EmailDetail]:
        """Fetch full email bodies by IDs."""
        if not email_ids:
            return []

        start = time.monotonic()
        logger.info("[DIAG] read_emails starting for %d email(s)", len(email_ids))

        headers = await self._get_headers()

        async def fetch_one(email_id: str) -> EmailDetail | None:
            try:
                return await self._fetch_email_detail(email_id, headers)
            except Exception as e:
                logger.warning("Failed to fetch email %s: %s", email_id, e)
                return None

        results_raw = await asyncio.gather(*[fetch_one(eid) for eid in email_ids])
        results = [r for r in results_raw if r is not None]

        elapsed = time.monotonic() - start
        logger.info(
            "[DIAG] read_emails fetched %d/%d emails in %.2fs",
            len(results),
            len(email_ids),
            elapsed,
        )
        return results

    async def _fetch_email_detail(
        self,
        email_id: str,
        headers: dict[str, str],
    ) -> EmailDetail | None:
        """Fetch a single email's full content using the content endpoint."""
        if ":" not in email_id:
            logger.warning("Invalid email ID format (no colon): %s", email_id)
            return None

        folder_id, message_id = email_id.split(":", 1)
        if not folder_id or not message_id:
            logger.warning("Invalid email ID format (empty parts): %s", email_id)
            return None

        account = await self._ensure_account()
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
        if text_body and "<" in text_body:
            text_body = strip_html(text_body)
        if len(text_body) > self._max_body_length:
            text_body = text_body[: self._max_body_length] + "\n\n[truncated]"

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

    async def draft_response(
        self,
        to_addresses: list[str],
        subject: str,
        content: str,
        cc_addresses: list[str] | None = None,
        bcc_addresses: list[str] | None = None,
        in_reply_to: str | None = None,
        mail_format: str = "plaintext",
    ) -> str | None:
        """Save an email draft to the Drafts folder."""
        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages"
        payload: dict[str, Any] = {
            "fromAddress": account.email_address,
            "toAddress": ",".join(to_addresses),
            "subject": subject,
            "content": content,
            "mode": "draft",
            "mailFormat": mail_format,
        }
        if cc_addresses:
            payload["ccAddress"] = ",".join(cc_addresses)
        if bcc_addresses:
            payload["bccAddress"] = ",".join(bcc_addresses)
        if in_reply_to:
            payload["inReplyTo"] = in_reply_to

        logger.info("Saving draft to %s: %s", to_addresses, subject)
        resp = await self._http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        draft_data = data.get("data", {})
        message_id = draft_data.get("messageId")
        if message_id:
            logger.info("Draft saved successfully: messageId=%s", message_id)
            return str(message_id)

        logger.warning("Draft saved but no messageId returned: %s", data)
        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
