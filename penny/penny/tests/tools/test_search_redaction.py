"""Tests for search query redaction of personal information."""

from penny.tools.builtin import SearchTool


class TestRedactQuery:
    """Unit tests for SearchTool._redact_query()."""

    def _make_tool(self, redact_terms: list[str]) -> SearchTool:
        """Create a SearchTool with mock Perplexity client and given redact terms."""
        tool = object.__new__(SearchTool)
        tool.redact_terms = redact_terms
        return tool

    def test_no_redact_terms(self):
        tool = self._make_tool([])
        assert tool._redact_query("weather in Toronto") == "weather in Toronto"

    def test_redacts_name_case_insensitive(self):
        tool = self._make_tool(["Jared"])
        assert tool._redact_query("Jared Toronto weather") == "Toronto weather"
        assert tool._redact_query("jared Toronto weather") == "Toronto weather"
        assert tool._redact_query("JARED Toronto weather") == "Toronto weather"

    def test_whole_word_only(self):
        tool = self._make_tool(["Ed"])
        assert tool._redact_query("Ed Sheeran music") == "Sheeran music"
        # "ed" inside "education" should not be redacted
        assert tool._redact_query("education news") == "education news"

    def test_collapses_whitespace(self):
        tool = self._make_tool(["Jared"])
        result = tool._redact_query("news for Jared in Toronto")
        assert "  " not in result
        assert result == "news for in Toronto"

    def test_multiple_occurrences(self):
        tool = self._make_tool(["Jared"])
        result = tool._redact_query("Jared likes what Jared likes")
        assert "Jared" not in result
        assert "jared" not in result.lower()

    def test_empty_terms_ignored(self):
        tool = self._make_tool(["", "Jared"])
        assert tool._redact_query("Jared news") == "news"

    def test_preserves_query_when_no_match(self):
        tool = self._make_tool(["Jared"])
        assert tool._redact_query("Toronto weather forecast") == "Toronto weather forecast"
