"""Model-refusal phrase detection.

Shared by the agent loop (which retries when the model emits a refusal
mid-conversation) and ``SendMessageTool`` (which refuses to dispatch a
refusal as if it were the user-facing reply).
"""

from __future__ import annotations

# Substrings that indicate the model has produced a refusal rather than
# substantive content.  Lowercase for case-insensitive matching.
REFUSAL_PHRASES = (
    "i can't",
    "i cannot",
    "i'm sorry",
    "i am sorry",
    "i'm unable",
    "i am unable",
    "i apologize",
    "as an ai",
    "as a language model",
)


def is_refusal(content: str) -> bool:
    """Return True if ``content`` contains any refusal phrase."""
    lower = content.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)
