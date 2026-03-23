"""Shared HTML text extraction utilities."""

from __future__ import annotations

import html.parser


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Simple HTML tag stripper."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(html_text: str) -> str:
    """Strip HTML tags and return plain text."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html_text)
    return extractor.get_text()
