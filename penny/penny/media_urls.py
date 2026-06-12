"""Media URLs — image references that travel inline through model text.

The browse tool stores each page image as a media table row and embeds a
reference URL (``https://media.penny.local/<id>``) in its result text.
Because the reference IS a URL, it rides the existing URL machinery —
the prompts already promote source-URL preservation, and the agent loop
already validates URLs against the context — with no media-specific
prompting or validation.  At channel egress the reserved host is
recognised, the URL is stripped from the outgoing text, and the binary
is attached to the message.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from penny.database.media_store import MediaStore

logger = logging.getLogger(__name__)

MEDIA_URL_HOST = "media.penny.local"

# Matches a media reference URL, with the optional "Image: " label the
# browse tool writes and any trailing punctuation the model may add.
MEDIA_URL_PATTERN = re.compile(rf"(?:Image: )?https://{re.escape(MEDIA_URL_HOST)}/(\d+)")

_DATA_URI_PATTERN = re.compile(r"^data:([\w.+-]+/[\w.+-]+);base64,(.+)$", re.DOTALL)


class ParsedDataUri(BaseModel):
    """Decoded contents of a `data:<mime>;base64,...` URI."""

    mime_type: str
    data: bytes


def format_media_url(media_id: int) -> str:
    """Render a media row id as its inline reference URL."""
    return f"https://{MEDIA_URL_HOST}/{media_id}"


def extract_media_ids(text: str) -> list[int]:
    """Return the media ids referenced in ``text``, ordered, deduplicated."""
    seen: set[int] = set()
    ids: list[int] = []
    for match in MEDIA_URL_PATTERN.finditer(text):
        media_id = int(match.group(1))
        if media_id not in seen:
            seen.add(media_id)
            ids.append(media_id)
    return ids


def strip_media_urls(text: str) -> str:
    """Remove all media reference URLs from ``text``, tidying blank lines."""
    stripped = MEDIA_URL_PATTERN.sub("", text)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def parse_data_uri(uri: str) -> ParsedDataUri | None:
    """Decode a base64 data URI, or None if ``uri`` isn't one.

    The browser extension delivers page images as ``data:<mime>;base64,...``
    strings (or empty).  Anything else is unexpected — callers log and skip.
    """
    match = _DATA_URI_PATTERN.match(uri)
    if match is None:
        return None
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except binascii.Error:
        return None
    return ParsedDataUri(mime_type=match.group(1), data=data)


def build_data_uri(mime_type: str, data: bytes) -> str:
    """Encode binary media back into a base64 data URI for channel delivery."""
    encoded = base64.b64encode(data).decode()
    return f"data:{mime_type};base64,{encoded}"


class ResolvedMedia(BaseModel):
    """Outgoing text with its media URLs stripped and resolved to data URIs."""

    content: str
    attachments: list[str]


def resolve_media_urls(media_store: MediaStore, content: str) -> ResolvedMedia:
    """Strip media reference URLs from outgoing text, loading each blob.

    Called at channel egress — the user-visible text never contains the
    reserved host, and the resolved attachments ride alongside the
    message.  References whose id has no media row are dropped with a
    warning.
    """
    ids = extract_media_ids(content)
    if not ids:
        return ResolvedMedia(content=content, attachments=[])
    attachments: list[str] = []
    for media_id in ids:
        row = media_store.get(media_id)
        if row is None:
            logger.warning("Dropping unknown media reference %s", format_media_url(media_id))
            continue
        attachments.append(build_data_uri(row.mime_type, row.data))
    return ResolvedMedia(content=strip_media_urls(content), attachments=attachments)
