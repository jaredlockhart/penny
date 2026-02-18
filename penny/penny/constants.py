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

    class EngagementType(StrEnum):
        """Type of user engagement with an entity."""

        EXPLICIT_STATEMENT = "explicit_statement"
        EMOJI_REACTION = "emoji_reaction"
        SEARCH_INITIATED = "search_initiated"
        FOLLOW_UP_QUESTION = "follow_up_question"
        LEARN_COMMAND = "learn_command"
        MESSAGE_MENTION = "message_mention"
        LIKE_COMMAND = "like_command"
        DISLIKE_COMMAND = "dislike_command"

    class EngagementValence(StrEnum):
        """Sentiment direction of an engagement."""

        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

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
    ENTITY_EXTRACTION_BATCH_LIMIT = 10
    ENTITY_CLEANING_BATCH_LIMIT = 200
    ENTITY_CLEANING_INTERVAL_SECONDS = 86400.0
    EMBEDDING_BACKFILL_BATCH_LIMIT = 50

    # Entity context injection constants
    ENTITY_CONTEXT_TOP_K = 5
    ENTITY_CONTEXT_THRESHOLD = 0.3
    ENTITY_CONTEXT_MAX_FACTS = 5
    KNOWLEDGE_SUFFICIENT_MIN_FACTS = 3
    KNOWLEDGE_SUFFICIENT_MIN_SCORE = 0.5

    # Engagement strength weights (0.0-1.0)
    ENGAGEMENT_STRENGTH_LEARN_COMMAND = 1.0
    ENGAGEMENT_STRENGTH_LIKE_COMMAND = 0.8
    ENGAGEMENT_STRENGTH_DISLIKE_COMMAND = 0.8
    ENGAGEMENT_STRENGTH_EXPLICIT_STATEMENT = 0.7
    ENGAGEMENT_STRENGTH_SEARCH_INITIATED = 0.6
    ENGAGEMENT_STRENGTH_FOLLOW_UP_QUESTION = 0.5
    ENGAGEMENT_STRENGTH_EMOJI_REACTION_PROACTIVE = 0.5
    ENGAGEMENT_STRENGTH_EMOJI_REACTION_NORMAL = 0.3
    ENGAGEMENT_STRENGTH_EMOJI_REACTION_PROACTIVE_NEGATIVE = 0.8
    ENGAGEMENT_STRENGTH_MESSAGE_MENTION = 0.2

    # Interest score recency decay half-life in days
    INTEREST_SCORE_HALF_LIFE_DAYS = 30.0
