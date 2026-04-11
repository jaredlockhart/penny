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


class ProgressEmoji(StrEnum):
    """Emojis used by ProgressTracker implementations to surface in-flight work.

    Channels that show progress as reactions on the user's message (e.g.
    SignalChannel) post one of these and morph between them as the agent's
    tool calls fire. Tools pick which one applies to their work via
    ``Tool.to_progress_emoji``.
    """

    THINKING = "\U0001f4ad"  # 💭 — initial state, before any tool calls
    SEARCHING = "\U0001f50d"  # 🔍 — running a text search
    READING = "\U0001f4d6"  # 📖 — reading a specific URL
    WORKING = "\u2699\ufe0f"  # ⚙️ — generic fallback for other tools


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
    BROWSE_ERROR_HEADER = "## browse error: "
    BROWSE_TITLE_PREFIX = "Title: "
    BROWSE_URL_PREFIX = "URL: "
    SECTION_SEPARATOR = "\n\n---\n\n"
    DISLIKE_FILTER_THRESHOLD = 0.8

    # Email command constants
    JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"

    # Zoho Mail API constants
    ZOHO_TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
    ZOHO_ACCOUNTS_URL = "https://mail.zoho.com/api/accounts"
    ZOHO_API_BASE = "https://mail.zoho.com/api"

    # Zoho Calendar API constants
    ZOHO_CALENDAR_API_BASE = "https://calendar.zoho.com/api/v1"

    # Zoho Projects API constants (v3)
    ZOHO_PROJECTS_API_BASE = "https://projectsapi.zoho.com/api/v3"

    # Test mode constants
    TEST_DB_PATH = Path("data/penny/penny-test.db")

    # Signal API connectivity validation
    SIGNAL_VALIDATE_MAX_ATTEMPTS = 12
    SIGNAL_VALIDATE_RETRY_DELAY = 5.0
    SIGNAL_VALIDATE_HTTP_TIMEOUT = 5.0

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
    # Minimum count of alphabetic characters for a model response to be
    # considered substantive. Catches garbage shapes — bare separators
    # (`---`), lone punctuation, emoji-only, runs of stars/dashes — without
    # enumerating them, while still allowing terse legit replies like "done"
    # or "yes". Anything below this is treated as EMPTY and retried.
    MIN_RESPONSE_LETTERS = 3
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

    # Related-messages retrieval constants
    #
    # Each candidate is scored as: adjusted = cosine_to_current - α * centrality
    # where centrality is the message's mean cosine to the rest of the corpus —
    # the centroid-magnet penalty that suppresses generic boilerplate which
    # would otherwise leak into every unrelated query. All values calibrated
    # empirically against the embeddinggemma corpus.
    RELATED_MESSAGES_CENTRALITY_PENALTY = 0.5
    # Cluster-strength gate: top_head_mean / top_sample_mean must exceed this
    # for any messages to be returned — separates real clusters from flat
    # noise plateaus. Calibrated in adjusted-score space.
    RELATED_MESSAGES_CLUSTER_GATE = 1.15
    # Cutoff is max(top_head_mean * RELATIVE_RATIO, ABSOLUTE_FLOOR). The
    # relative band adapts cluster width to cluster height; the absolute floor
    # is the empirical noise ceiling below which adjusted scores are
    # statistically indistinguishable from random.
    RELATED_MESSAGES_RELATIVE_RATIO = 0.85
    RELATED_MESSAGES_ABSOLUTE_FLOOR = 0.25
    # Number of top candidates averaged to estimate the cluster center
    # (numerator of the gate ratio).
    RELATED_MESSAGES_GATE_HEAD_SIZE = 5
    # Number of top candidates averaged to estimate the broader noise floor
    # (denominator of the gate ratio). Also doubles as the cold-start
    # threshold — below this we skip the gate and use just the absolute floor.
    RELATED_MESSAGES_GATE_SAMPLE_SIZE = 20
    # After scoring + cutoff selects hits, expand each hit by ±N minutes of
    # surrounding user messages. Captures conversational follow-ups that have
    # no entity overlap with the current message ("yeah exactly i can't wait
    # to try it") but live in the same conversation as a real hit.
    RELATED_MESSAGES_NEIGHBOR_WINDOW_MINUTES = 5
