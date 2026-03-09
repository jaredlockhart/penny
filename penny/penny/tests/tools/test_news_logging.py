"""Tests for NewsTool exception logging."""

import logging

import pytest

from penny.tools.news import NewsTool


class TestNewsToolLogging:
    """Verify that NewsTool logs exceptions with full traceback info."""

    @pytest.mark.asyncio
    async def test_exc_info_logged_on_empty_message_exception(self, caplog):
        """Exception with no message must log exc_info so type and traceback are captured."""
        tool = NewsTool(api_key="invalid-key")

        # Patch _call_api to raise an exception with no message
        async def raise_no_message(**kwargs):
            raise RuntimeError

        tool._call_api = raise_no_message  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger="penny.tools.news"):
            result = await tool.search(query_terms=["test"])

        assert result == []
        # exc_info=True causes the record to include exc_info tuple
        assert any(r.exc_info is not None for r in caplog.records), (
            "Expected exc_info to be captured — logger.error() must pass exc_info=True"
        )
