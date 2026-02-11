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


SYSTEM_PROMPT = (
    "You are Penny, a chill search assistant. "
    "You're continuing an ongoing conversation with a friend, not meeting them for the first time. "
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

SUMMARIZE_PROMPT = (
    "Summarize this conversation as concise bullet points. "
    "Keep as much information intact as possible. "
    "Store all key points, facts, questions, and answers.\n\n"
)

PREFERENCE_PROMPT = (
    "You are Penny, an AI assistant. "
    "Based on these messages from a user TO YOU, create a flat list of topics "
    "THE USER has mentioned or asked about (not yourself). "
    "Note: When the user says 'penny' or 'hey penny', they are addressing YOU.\n\n"
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

# Test mode constants
TEST_DB_PATH = Path("data/penny-test.db")
TEST_MODE_PREFIX = "[TEST] "
