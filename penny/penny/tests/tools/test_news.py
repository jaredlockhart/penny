"""Tests for NewsTool error handling."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from penny.tools.news import NewsTool


@pytest.mark.asyncio
async def test_fetch_articles_logs_exc_info_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Exceptions with no message are still fully logged via exc_info=True."""

    class _SilentError(Exception):
        def __str__(self) -> str:
            return ""

    tool = NewsTool(api_key="test-key")

    with (
        patch.object(tool, "_call_api", new_callable=AsyncMock, side_effect=_SilentError()),
        caplog.at_level(logging.ERROR, logger="penny.tools.news"),
    ):
        result = await tool._fetch_articles("test", from_date=None)

    assert result == []
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert "Unexpected error fetching news" in record.message
    # exc_info=True means the exception info is captured on the log record
    assert record.exc_info is not None
    assert record.exc_info[0] is _SilentError
