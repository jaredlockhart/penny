"""Constants and prompts for Penny agent."""

from enum import Enum


class MessageDirection(str, Enum):
    """Direction of a logged message."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"


SYSTEM_PROMPT = (
    "You are Penny, a helpful AI assistant. "
    "You MUST use the search tool for every message to research your answer. "
    "Never answer from your own knowledge alone - always search first, then respond "
    "based on the search results. "
    "Only use plain text - no markdown, no bullet points, no formatting. "
    "Only use lowercase. "
    "Speak casually. "
    "End every response with an emoji."
)

# Search tool constants
PERPLEXITY_PRESET = "pro-search"
NO_RESULTS_TEXT = "No results found"
IMAGE_MAX_RESULTS = 3
IMAGE_DOWNLOAD_TIMEOUT = 15.0

SUMMARIZE_PROMPT = (
    "Summarize this conversation as concise bullet points. "
    "Keep as much information intact as possible. "
    "Store all key points, facts, questions, and answers.\n\n"
)
