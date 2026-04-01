"""Post-processing for web page content returned by the browser extension.

Strips structural cruft (proxy images, navigation, licensing boilerplate)
that Turndown faithfully converts to markdown but that wastes model context.

Patterns are grouped by source:
  - Kagi search result pages (proxy images, image grids, Openverse)
  - Cross-site (tracking pixels, JSON blobs, nav headers, long auth URLs)

To find new patterns: check the logs for large browse_url results, compare
the raw and cleaned sizes, and look for repetitive non-content lines.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Kagi-specific ─────────────────────────────────────────────────────────

# Kagi image grid section starts with [Images] or [Videos] links
_KAGI_IMAGE_VIDEO_HEADER = re.compile(r"^\[(Images|Videos)]\(https://kagi\.com/")

# Bare domain attribution under Kagi thumbnail images: [www.example.com](https://...)
_KAGI_DOMAIN_ATTRIBUTION = re.compile(r"^\[www\.\S+]\(https://")

# Openverse metadata labels in the Kagi image lightbox
_KAGI_METADATA_LABELS = frozenset({"Size", "Type", "Uploaded", "Aspect"})

# ── Cross-site ────────────────────────────────────────────────────────────

# Empty markdown links with no display text: [](https://...)
_EMPTY_LINK = re.compile(r"^\[]\(https?://")

# Image dimension lines: "1920 x 1080", "980 x 980"
_DIMENSIONS = re.compile(r"^\d{2,5}\s*x\s*\d{2,5}$")

# Tracking pixel images (Amazon, analytics, etc.)
_TRACKING_PIXEL = re.compile(r"!\[.*?]\(https?://[^\)]*transparent-pixel[^\)]*\)")

# Inline JSON blobs (Amazon pricing, structured data) — line is >{...}<
_JSON_BLOB = re.compile(r'^["\']?\{".{50,}\}["\']?$')

# Links where the URL portion exceeds 300 chars (auth/signin, tracking)
_LONG_URL_LINK = re.compile(r"^\[.*?\]\((https?://[^\)]{300,})\)$")

# Navigation boilerplate lines
_NAV_BOILERPLATE = frozenset(
    {
        "Skip to main content",
        "Skip to search results",
        "Skip to content",
        "Skip to navigation",
        "Hide Filters",
        "Reset filters",
        "Article continues below",
        "Show Filters",
        "View Image Gallery",
    }
)

# Repeated category label pattern from listing pages
# e.g. "NewsCategory: News.|TechnologyCategory: Technology."
_CATEGORY_LABELS = re.compile(r"^(\w+Category:\s*\w+\.\|?){2,}$")


def clean_browser_content(text: str) -> str:
    """Remove structural cruft from browser-extracted markdown content."""
    lines = text.split("\n")
    cleaned: list[str] = []
    skipping_kagi_images = False

    for line in lines:
        stripped = line.strip()

        if _should_skip_line(stripped):
            continue

        # Kagi image/video grid — skip everything until next search result
        if _KAGI_IMAGE_VIDEO_HEADER.match(stripped):
            skipping_kagi_images = True
            continue

        if skipping_kagi_images:
            if stripped.startswith("### ") or stripped.startswith("![Favicon"):
                skipping_kagi_images = False
                if stripped.startswith("![Favicon"):
                    continue
            else:
                continue

        cleaned.append(line)

    result = "\n".join(cleaned)
    # Collapse runs of 3+ blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    removed = len(text) - len(result)
    if removed > 0:
        logger.debug(
            "content_cleaning: %d -> %d chars (removed %d, %.0f%%)",
            len(text),
            len(result),
            removed,
            removed / len(text) * 100,
        )

    return result


def _should_skip_line(stripped: str) -> bool:
    """Return True if this line is structural cruft that should be removed."""
    # ── Kagi-specific ──
    if stripped.startswith("![Favicon"):
        return True
    if stripped.startswith(("[![](https://p.kagi.com/proxy/", "![](https://p.kagi.com/proxy/")):
        return True
    if _KAGI_DOMAIN_ATTRIBUTION.match(stripped):
        return True
    if stripped in _KAGI_METADATA_LABELS:
        return True
    if "Made with [Openverse]" in stripped:
        return True
    if "Upload time and" in stripped:
        return True
    if stripped.startswith(("[View Image]", "[Download]", "[Visit Page]")):
        return True
    if "Loading source..." in stripped or "Loading license..." in stripped:
        return True
    if stripped.startswith("Report to [OpenVerse]"):
        return True

    # ── Cross-site ──

    # Empty links
    if _EMPTY_LINK.match(stripped):
        return True

    # Image dimension lines
    if _DIMENSIONS.match(stripped):
        return True

    # Tracking pixel images
    if _TRACKING_PIXEL.match(stripped):
        return True

    # Inline JSON blobs
    if _JSON_BLOB.match(stripped):
        return True

    # Links with extremely long URLs (auth/signin/tracking)
    if _LONG_URL_LINK.match(stripped):
        return True

    # Navigation boilerplate
    if stripped in _NAV_BOILERPLATE:
        return True

    # Repeated category labels from listing pages
    return bool(_CATEGORY_LABELS.match(stripped))
