"""Unit tests for Signal channel formatting (prepare_outgoing, _table_to_bullets)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from penny.channels.signal import SignalChannel


@pytest.fixture
def channel():
    """Minimal SignalChannel for testing formatting methods."""
    return SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+15551234567",
        message_agent=MagicMock(),
        db=MagicMock(),
    )


class TestTableToBullets:
    """Tests for _table_to_bullets classmethod."""

    def test_simple_table(self):
        text = (
            "| Model | Price | Type |\n"
            "|-------|-------|------|\n"
            "| Foo   | $100  | Basic |\n"
            "| Bar   | $200  | Pro   |"
        )
        result = SignalChannel._table_to_bullets(text)
        assert result == (
            "**Foo**\n"
            "  \u2022 **Price**: $100\n"
            "  \u2022 **Type**: Basic\n"
            "\n"
            "**Bar**\n"
            "  \u2022 **Price**: $200\n"
            "  \u2022 **Type**: Pro\n"
        )

    def test_preserves_non_table_text(self):
        text = "Hello world\nNo tables here"
        assert SignalChannel._table_to_bullets(text) == "Hello world\nNo tables here"

    def test_table_with_surrounding_text(self):
        text = (
            "Here are the results:\n\n"
            "| Name | Score |\n"
            "|------|-------|\n"
            "| Alice | 95   |\n"
            "\nEnd of results."
        )
        result = SignalChannel._table_to_bullets(text)
        assert result == (
            "Here are the results:\n\n"
            "**Alice**\n"
            "  \u2022 **Score**: 95\n"
            "\nEnd of results."
        )

    def test_strips_bold_from_first_column(self):
        text = (
            "| Speaker | Price |\n"
            "|---------|-------|\n"
            "| **Polk** | $100 |"
        )
        result = SignalChannel._table_to_bullets(text)
        assert result == (
            "**Polk**\n"
            "  \u2022 **Price**: $100\n"
        )

    def test_empty_cells_skipped(self):
        text = (
            "| Name | A | B |\n"
            "|------|---|---|\n"
            "| Foo  | 1 |   |"
        )
        result = SignalChannel._table_to_bullets(text)
        assert result == (
            "**Foo**\n"
            "  \u2022 **A**: 1\n"
        )


class TestPrepareOutgoing:
    """Tests for prepare_outgoing instance method."""

    def test_strips_heading_h1(self, channel):
        assert channel.prepare_outgoing("# Top") == "Top"

    def test_strips_heading_h2(self, channel):
        assert channel.prepare_outgoing("## Hello") == "Hello"

    def test_strips_heading_h3(self, channel):
        assert channel.prepare_outgoing("### Sub heading") == "Sub heading"

    def test_preserves_bold(self, channel):
        assert channel.prepare_outgoing("This is **bold** text") == "This is **bold** text"

    def test_preserves_italic(self, channel):
        assert channel.prepare_outgoing("This is *italic* text") == "This is *italic* text"

    def test_converts_double_tilde_to_single(self, channel):
        assert channel.prepare_outgoing("This is ~~struck~~ text") == "This is ~struck~ text"

    def test_escapes_stray_tilde(self, channel):
        assert channel.prepare_outgoing("About ~50 items") == "About \u223c50 items"

    def test_removes_horizontal_rule(self, channel):
        assert channel.prepare_outgoing("Before\n\n---\n\nAfter") == "Before\n\nAfter"

    def test_removes_long_horizontal_rule(self, channel):
        assert channel.prepare_outgoing("Before\n\n----------\n\nAfter") == "Before\n\nAfter"

    def test_table_separator_not_stripped(self, channel):
        """Table separator rows with pipes are converted to bullets, not stripped."""
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |"
        )
        assert channel.prepare_outgoing(text) == (
            "**1**\n"
            "  \u2022 **B**: 2"
        )

    def test_converts_markdown_links(self, channel):
        assert (
            channel.prepare_outgoing("[Click here](https://example.com)")
            == "Click here (https://example.com)"
        )

    def test_collapses_blank_lines(self, channel):
        assert channel.prepare_outgoing("A\n\n\n\n\nB") == "A\n\nB"

    def test_removes_stray_footnote_asterisk(self, channel):
        assert channel.prepare_outgoing(
            "Price: $950 CAD*\n\n- **Details:** info"
        ) == "Price: $950 CAD\n\n- **Details:** info"

    def test_stray_asterisk_does_not_break_bold(self, channel):
        assert channel.prepare_outgoing(
            "Item*\n- **Price:** $100\n- **Type:** Basic"
        ) == "Item\n- **Price:** $100\n- **Type:** Basic"

    def test_stray_asterisk_preserves_italic(self, channel):
        assert channel.prepare_outgoing(
            "Note*\n\n*This is italic text.*"
        ) == "Note\n\n*This is italic text.*"

    def test_no_stray_asterisks_is_noop(self, channel):
        assert (
            channel.prepare_outgoing("**Bold** and *italic* text")
            == "**Bold** and *italic* text"
        )

    def test_full_research_report(self, channel):
        text = (
            "## Top Speakers\n\n"
            "### 1. Polk R200\n"
            "- **Price:** $1,950\n"
            "- **Pros**\n"
            "  - Deep bass\n\n"
            "---\n\n"
            "### 2. KEF LS50*\n"
            "- **Price:** $1,970\n\n"
            "| Model | Price |\n"
            "|-------|-------|\n"
            "| Polk  | $1,950 |\n"
            "| KEF   | $1,970 |\n\n"
            "*All prices approximate.*"
        )
        assert channel.prepare_outgoing(text) == (
            "Top Speakers\n\n"
            "1. Polk R200\n"
            "- **Price:** $1,950\n"
            "- **Pros**\n"
            "  - Deep bass\n\n"
            "2. KEF LS50\n"
            "- **Price:** $1,970\n\n"
            "**Polk**\n"
            "  \u2022 **Price**: $1,950\n\n"
            "**KEF**\n"
            "  \u2022 **Price**: $1,970\n\n"
            "*All prices approximate.*"
        )
