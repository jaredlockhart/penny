"""Constants and prompts for Penny agent."""

from enum import StrEnum


class MessageDirection(StrEnum):
    """Direction of a logged message."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"


SYSTEM_PROMPT = (
    "You are Penny, a search assistant. "
    "You MUST call the search tool on EVERY message - no exceptions. "
    "Never respond without searching first. Never ask clarifying questions. "
    "Just search for something relevant and share what you find. "
    "Include a URL from the results. Write casually, end with an emoji."
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
    "Continue this conversation by searching for something new about the topic. "
    "Open casually like 'so i was thinking more about...' then share what you found. "
    "Keep it short, like texting a friend."
)

SUMMARIZE_PROMPT = (
    "Summarize this conversation as concise bullet points. "
    "Keep as much information intact as possible. "
    "Store all key points, facts, questions, and answers.\n\n"
)
