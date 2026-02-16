"""Constants for Penny agent."""

from enum import StrEnum
from pathlib import Path


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
NO_RESULTS_TEXT = "No results found"
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
EMAIL_NO_QUERY_TEXT = "Please ask a question about your email. Usage: /email <question>"

# Schedule command response messages
SCHEDULE_NO_TASKS = "You don't have any scheduled tasks yet 📅"
SCHEDULE_NEED_TIMEZONE = (
    "I need to know your timezone first. Send me your location or tell me your city 📍"
)
SCHEDULE_PARSE_ERROR = (
    "Sorry, I couldn't understand that schedule format. "
    "Try something like: /schedule daily 9am what's the news?"
)
SCHEDULE_INVALID_CRON = (
    "Sorry, I couldn't figure out the timing. "
    "Try something like: /schedule daily 9am what's the news?"
)
SCHEDULE_DELETED_NO_REMAINING = "No more scheduled tasks"
SCHEDULE_STILL_SCHEDULED = "Still scheduled:"
SCHEDULE_INVALID_NUMBER = "Invalid schedule number: {number}"
SCHEDULE_NO_SCHEDULE_WITH_NUMBER = "No schedule with number {number}"
SCHEDULE_DELETED_PREFIX = "Deleted '{timing} {prompt}' ✅"
SCHEDULE_ADDED = "Added {timing}: {prompt} ✅"

# Test mode constants
TEST_DB_PATH = Path("data/penny-test.db")
TEST_MODE_PREFIX = "[TEST] "

# GitHub constants
GITHUB_REPO_OWNER = "jaredlockhart"
GITHUB_REPO_NAME = "penny"

# Vision constants
VISION_NOT_CONFIGURED_MESSAGE = (
    "I can see you sent an image but I don't have vision configured right now."
)
VISION_SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
VISION_IMAGE_CONTEXT = "user said '{user_text}' and included an image of: {caption}"
VISION_IMAGE_ONLY_CONTEXT = "user sent an image of: {caption}"

# Personality command response messages
PERSONALITY_NO_CUSTOM = "No custom personality set. Using default Penny personality."
PERSONALITY_RESET_SUCCESS = "Ok, personality reset to default ✅"
PERSONALITY_RESET_NOT_SET = "No custom personality was set."
PERSONALITY_UPDATE_SUCCESS = "Ok, personality updated ✅"
