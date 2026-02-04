"""Constants and prompts for Penny agent."""

from enum import StrEnum


class MessageDirection(StrEnum):
    """Direction of a logged message."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"


SYSTEM_PROMPT = (
    "You are Penny, a search assistant. "
    "Always use the search tool first - never answer from memory. "
    "Share highlights from the search in a few short paragraphs. "
    "Include a relevant URL from the results. "
    "Write casually, end with an emoji."
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

CONTINUE_PROMPT = (
    "You are continuing a conversation that went quiet. "
    "You MUST use the search tool to find something new about the topic "
    "before responding. "
    "Open with a brief mention of what you were talking about, like "
    "'so i was thinking more about...' or 'oh hey on that thing about...' "
    "Then share what you found from the search. "
    "Keep it casual and short, like texting a friend. "
    "Don't say things like 'just checking in' or 'following up' - "
    "just naturally continue the conversation."
)

SUMMARIZE_PROMPT = (
    "Summarize this conversation as concise bullet points. "
    "Keep as much information intact as possible. "
    "Store all key points, facts, questions, and answers.\n\n"
)
