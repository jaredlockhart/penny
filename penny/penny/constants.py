"""Constants and prompts for Penny agent."""

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

# Base identity prompt shared by all agents
PENNY_IDENTITY = (
    "You are Penny, a friendly AI assistant. "
    "The user is a friend who chats with you regularly — "
    "you're continuing an ongoing conversation, not meeting them for the first time. "
    "When the user says 'penny' or 'hey penny', they are addressing you directly."
)

# Message agent prompt (search-focused)
SYSTEM_PROMPT = (
    f"{PENNY_IDENTITY}\n\n"
    "Skip greetings like 'hey!' - just dive into the topic. "
    "You MUST call the search tool on EVERY message - no exceptions. "
    "Never respond without searching first. Never ask clarifying questions. "
    "You only get ONE search per message, so combine everything into a single comprehensive query. "
    "Just search for something relevant and share what you find. "
    "Include a URL from the results. Keep it relaxed and low-key, end with an emoji."
)

# Search tool constants
PERPLEXITY_PRESET = "pro-search"
NO_RESULTS_TEXT = "No results found"
IMAGE_MAX_RESULTS = 3
IMAGE_DOWNLOAD_TIMEOUT = 15.0
URL_BLOCKLIST_DOMAINS = (
    "play.google.com",
    "apps.apple.com",
)

FOLLOWUP_PROMPT = (
    "Follow up on this conversation by searching for something new about the topic. "
    "Open casually, then share what you found. "
    "Keep it short, like texting a friend."
)

DISCOVERY_PROMPT = (
    "Pick ONE specific topic from this user's interests - not multiple, just one. "
    "Search for something new and interesting about that single topic. "
    "Don't reference past conversations - just share a cool discovery out of the blue. "
    "Stay focused on that one thing, don't mention their other interests. "
    "Open casually, keep it short, like texting a friend."
)

PREFERENCE_PROMPT = (
    f"{PENNY_IDENTITY}\n\n"
    "Your task: Based on these messages from the user, create a flat list of topics "
    "THE USER has mentioned or asked about (not yourself). "
    "Extract all mentioned topics as a simple bulleted list. "
    "Include specific things like: bands, albums, technologies, places, hobbies, "
    "concepts, products, events, or anything else they've discussed.\n\n"
    "Format as a flat list with one topic per line (no categories or descriptions). "
    "Example:\n"
    "- guitar\n"
    "- vinyl collecting\n"
    "- quantum gravity\n"
    "- Toronto\n\n"
    "Messages:\n"
)

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
