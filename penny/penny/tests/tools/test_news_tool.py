"""Tests for NewsTool error handling."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from penny.tools.news import NewsTool


@pytest.mark.asyncio
async def test_unexpected_error_logs_exception_type(caplog: pytest.LogCaptureFixture) -> None:
    """Empty-message exceptions like ConnectError should still log the exception class name."""
    tool = NewsTool(api_key="test-key")

    # httpx.ConnectError with no message serializes str(e) as ""
    empty_msg_error = httpx.ConnectError("")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=empty_msg_error)
        mock_client_cls.return_value = mock_client

        with caplog.at_level(logging.ERROR, logger="penny.tools.news"):
            articles = await tool.search(query_terms=["test"])

    assert articles == []
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "ConnectError" in record.getMessage()


@pytest.mark.asyncio
async def test_http_status_error_logged_separately(caplog: pytest.LogCaptureFixture) -> None:
    """HTTP status errors are handled by the specific httpx.HTTPStatusError branch."""
    tool = NewsTool(api_key="test-key")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "rate limited"
    http_error = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=mock_response)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock(side_effect=http_error)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        with caplog.at_level(logging.ERROR, logger="penny.tools.news"):
            articles = await tool.search(query_terms=["test"])

    assert articles == []
    assert len(caplog.records) == 1
    assert "429" in caplog.records[0].getMessage()
