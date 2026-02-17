"""Constants for Penny agent."""

from enum import StrEnum
from pathlib import Path


class PennyConstants:
    """All constants for the Penny agent."""

    class MessageDirection(StrEnum):
        """Direction of a logged message."""

        INCOMING = "incoming"
        OUTGOING = "outgoing"

    class PreferenceType(StrEnum):
        """Type of user preference."""

        LIKE = "like"
        DISLIKE = "dislike"

    # Reaction emoji mappings for sentiment analysis
    LIKE_REACTIONS = ("❤️", "👍", "😆")
    DISLIKE_REACTIONS = ("😠", "👎", "😢")

    # Max messages/reactions the PreferenceAgent processes per user per wake cycle.
    # Keeps LLM prompts small; remaining items are processed on subsequent invocations.
    PREFERENCE_BATCH_LIMIT = 20

    # Search tool constants
    PERPLEXITY_PRESET = "pro-search"
    IMAGE_MAX_RESULTS = 3
    IMAGE_DOWNLOAD_TIMEOUT = 15.0
    URL_BLOCKLIST_DOMAINS = (
        "play.google.com",
        "apps.apple.com",
    )

    RESEARCH_FOCUS_TIMEOUT_SECONDS = 300

    # Email command constants
    JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"
    JMAP_REQUEST_TIMEOUT = 30.0
    EMAIL_BODY_MAX_LENGTH = 4000

    # Test mode constants
    TEST_DB_PATH = Path("data/penny-test.db")

    # GitHub constants
    GITHUB_REPO_OWNER = "jaredlockhart"
    GITHUB_REPO_NAME = "penny"

    # Vision constants
    VISION_SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")

    # Entity knowledge base constants
    class EntityType(StrEnum):
        """Type of entity in the knowledge base."""

        PRODUCT = "product"
        PERSON = "person"
        PLACE = "place"
        CONCEPT = "concept"
        ORGANIZATION = "organization"
        EVENT = "event"

    ENTITY_EXTRACTION_BATCH_LIMIT = 10
    ENTITY_CONTEXT_MAX_ENTITIES = 10
    ENTITY_CONTEXT_MAX_FACTS = 5  # per entity in context injection
