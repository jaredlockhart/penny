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
        SEARCH_DISCOVERY = "search_discovery"

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

    # Max messages/reactions the extraction pipeline processes per user per wake cycle.
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

    # Email command constants
    JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"
    JMAP_REQUEST_TIMEOUT = 30.0
    EMAIL_BODY_MAX_LENGTH = 4000

    # Test mode constants
    TEST_DB_PATH = Path("data/penny/penny-test.db")

    # GitHub constants
    GITHUB_REPO_OWNER = "jaredlockhart"
    GITHUB_REPO_NAME = "penny"

    # Vision constants
    VISION_SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")

    # Entity knowledge base constants
    ENTITY_EXTRACTION_BATCH_LIMIT = 10
    EMBEDDING_BACKFILL_BATCH_LIMIT = 50

    # Message pre-filter constants for extraction pipeline
    MIN_EXTRACTION_MESSAGE_LENGTH = 20

    # Fact deduplication via embedding similarity
    FACT_DEDUP_SIMILARITY_THRESHOLD = 0.85

    # Preference-to-entity linking via embedding similarity
    PREFERENCE_ENTITY_LINK_THRESHOLD = 0.5

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

    # Maximum entities shown by /interests command
    INTERESTS_DISPLAY_LIMIT = 20

    # Learn loop constants
    LEARN_ENRICHMENT_FACT_THRESHOLD = 5  # Below this → enrichment mode
    LEARN_STALENESS_DAYS = 7.0  # Days until facts are considered stale
    LEARN_MIN_INTEREST_SCORE = 0.1  # Minimum interest to consider
    LEARN_RECENT_DAYS = 1.0  # Skip entity if verified within this window

    # Entity name validation
    ENTITY_NAME_SEMANTIC_THRESHOLD = 0.58

    # Entity pre-filter for extraction pipeline
    ENTITY_PREFILTER_SIMILARITY_THRESHOLD = 0.2
    ENTITY_PREFILTER_MIN_COUNT = 20

    # Entity dedup at insertion time (TCR + embedding dual threshold)
    ENTITY_DEDUP_TCR_THRESHOLD = 0.75
    ENTITY_DEDUP_EMBEDDING_THRESHOLD = 0.85

    # Fact discovery notification backoff and quality gate
    FACT_NOTIFICATION_INITIAL_BACKOFF = 60.0  # seconds; first backoff after unanswered cycle
    FACT_NOTIFICATION_MAX_BACKOFF = 3600.0  # seconds; cap at 1 hour
    FACT_NOTIFICATION_MIN_LENGTH = 75  # skip near-empty model outputs

    # LearnPrompt defaults
    LEARN_PROMPT_DEFAULT_SEARCHES = 5
    LEARN_STATUS_DISPLAY_LIMIT = 10
