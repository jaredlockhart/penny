"""Constants for Penny agent."""

from enum import StrEnum
from pathlib import Path


class ChannelType(StrEnum):
    """Communication channel types."""

    SIGNAL = "signal"
    DISCORD = "discord"
    BROWSER = "browser"


class DomainPermissionValue(StrEnum):
    """Domain access permission states."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"


class ValidationReason(StrEnum):
    """Reasons a model response failed validation."""

    XML = "xml"
    EMPTY = "empty"
    REFUSAL = "refusal"
    HALLUCINATED_URLS = "hallucinated_urls"


class PennyConstants:
    """All constants for the Penny agent."""

    class MessageDirection(StrEnum):
        """Direction of a logged message."""

        INCOMING = "incoming"
        OUTGOING = "outgoing"

    class SearchTrigger(StrEnum):
        """What triggered a search."""

        USER_MESSAGE = "user_message"
        PENNY_ENRICHMENT = "penny_enrichment"

    # Browse tool constants
    URL_BLOCKLIST_DOMAINS = (
        "play.google.com",
        "apps.apple.com",
    )
    BROWSE_RETRIES = 4
    BROWSE_RETRY_DELAY = 1.0
    MAX_SEARCH_LINKS = 10
    BROWSE_SEARCH_HEADER = "## search: "
    BROWSE_PAGE_HEADER = "## browse: "
    BROWSE_TITLE_PREFIX = "Title: "
    SECTION_SEPARATOR = "\n\n---\n\n"
    DISLIKE_FILTER_THRESHOLD = 0.8
    KNOWLEDGE_WATERMARK_KEY = "knowledge_extraction_watermark"

    # Email command constants
    JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"

    # Zoho Mail API constants
    ZOHO_TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
    ZOHO_ACCOUNTS_URL = "https://mail.zoho.com/api/accounts"
    ZOHO_API_BASE = "https://mail.zoho.com/api"

    # Test mode constants
    TEST_DB_PATH = Path("data/penny/penny-test.db")

    # GitHub constants
    GITHUB_REPO_OWNER = "jaredlockhart"
    GITHUB_REPO_NAME = "penny"

    class PreferenceValence(StrEnum):
        """Valence of a user preference."""

        POSITIVE = "positive"
        NEGATIVE = "negative"

    class PreferenceSource(StrEnum):
        """How a preference was created."""

        MANUAL = "manual"
        EXTRACTED = "extracted"

    POSITIVE_REACTION_EMOJIS = frozenset(
        {
            "\U0001f44d",  # 👍
            "\u2764\ufe0f",  # ❤️
            "\U0001f525",  # 🔥
            "\U0001f44f",  # 👏
            "\U0001f60d",  # 😍
            "\U0001f64c",  # 🙌
            "\U0001f4af",  # 💯
            "\u2b50",  # ⭐
            "\U0001f60a",  # 😊
            "\U0001f389",  # 🎉
            "\U0001f4aa",  # 💪
            "\u2705",  # ✅
            "\U0001f929",  # 🤩
        }
    )

    NEGATIVE_REACTION_EMOJIS = frozenset(
        {
            "\U0001f44e",  # 👎
            "\U0001f621",  # 😡
            "\U0001f92e",  # 🤮
            "\U0001f4a9",  # 💩
            "\U0001f624",  # 😤
            "\u274c",  # ❌
            "\U0001f61e",  # 😞
            "\U0001f612",  # 😒
            "\U0001f644",  # 🙄
        }
    )

    # Vision constants
    VISION_SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")

    # Agent loop constants
    VISION_MAX_STEPS = 1
    CHECKIN_MAX_STEPS = 1
    RESPONSE_VALIDATION_RETRIES = 5
    TOOL_FAILURE_ABORT_THRESHOLD = 2
    THOUGHT_CONTEXT_LIMIT = 10
    PREFERRED_POOL_SIZE = 5

    # Thinking constants
    MIN_THOUGHT_WORDS = 50
    SUMMARY_URL_RETRIES = 2

    # Notify constants
    CHECKIN_ACTIVE_WINDOW = 1800
    CHECKIN_COOLDOWN_SECONDS = 86400
    THOUGHT_TOPIC_COOLDOWN_SECONDS = 86400
    NOTIFY_URL_RETRIES = 2

    # Browser channel constants
    TOOL_REQUEST_TIMEOUT = 60.0
    PERMISSION_PROMPT_TIMEOUT = 60.0
    MAX_PAGE_CONTENT_CHARS = 100_000
