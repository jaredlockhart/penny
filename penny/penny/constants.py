"""Constants for Penny agent."""

from enum import StrEnum
from pathlib import Path


class PennyConstants:
    """All constants for the Penny agent."""

    class MessageDirection(StrEnum):
        """Direction of a logged message."""

        INCOMING = "incoming"
        OUTGOING = "outgoing"

    class EngagementType(StrEnum):
        """Type of user engagement with an entity."""

        EXPLICIT_STATEMENT = "explicit_statement"
        EMOJI_REACTION = "emoji_reaction"
        USER_SEARCH = "user_search"
        FOLLOW_UP_QUESTION = "follow_up_question"
        MESSAGE_MENTION = "message_mention"
        SEARCH_DISCOVERY = "search_discovery"
        NOTIFICATION_IGNORED = "notification_ignored"

    # Engagement types that count for notification entity scoring.
    # Excludes user_search and search_discovery (noisy batch signals from /learn)
    # but includes notification_ignored (auto-tuning soft-veto).
    NOTIFICATION_ENGAGEMENT_TYPES = frozenset(
        {
            EngagementType.EMOJI_REACTION,
            EngagementType.EXPLICIT_STATEMENT,
            EngagementType.FOLLOW_UP_QUESTION,
            EngagementType.MESSAGE_MENTION,
            EngagementType.NOTIFICATION_IGNORED,
        }
    )

    class EngagementValence(StrEnum):
        """Sentiment direction of an engagement."""

        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    class SearchTrigger(StrEnum):
        """What triggered a search."""

        USER_MESSAGE = "user_message"
        LEARN_COMMAND = "learn_command"
        PENNY_ENRICHMENT = "penny_enrichment"

    class LearnPromptStatus(StrEnum):
        """Status of a LearnPrompt lifecycle."""

        ACTIVE = "active"
        COMPLETED = "completed"

    # Reaction emoji mappings for sentiment analysis
    LIKE_REACTIONS = ("❤️", "👍", "😆")
    DISLIKE_REACTIONS = ("😠", "👎", "😢")

    # Search tool constants
    PERPLEXITY_PRESET = "pro-search"
    URL_BLOCKLIST_DOMAINS = (
        "play.google.com",
        "apps.apple.com",
    )

    # Email command constants
    JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"

    # Test mode constants
    TEST_DB_PATH = Path("data/penny/penny-test.db")

    # GitHub constants
    GITHUB_REPO_OWNER = "jaredlockhart"
    GITHUB_REPO_NAME = "penny"

    class EventSourceType(StrEnum):
        """Source that produced an event."""

        NEWS_API = "news_api"
        SEARCH = "search"

    class FollowPromptStatus(StrEnum):
        """Status of a FollowPrompt lifecycle."""

        ACTIVE = "active"
        CANCELLED = "cancelled"

    # Vision constants
    VISION_SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
