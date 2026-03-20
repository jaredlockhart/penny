"""Constants for Penny agent."""

from enum import StrEnum
from pathlib import Path


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

    # Search tool constants
    PERPLEXITY_PRESET = "pro-search"
    URL_BLOCKLIST_DOMAINS = (
        "play.google.com",
        "apps.apple.com",
    )

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

    class HistoryDuration(StrEnum):
        """Duration granularity for conversation history summaries."""

        DAILY = "daily"
        WEEKLY = "weekly"
        MONTHLY = "monthly"

    class PreferenceValence(StrEnum):
        """Valence of a user preference."""

        POSITIVE = "positive"
        NEGATIVE = "negative"

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
