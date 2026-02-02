"""Constants for Penny agent."""


class SystemPrompts:
    """System prompts for different contexts."""

    MESSAGE_HANDLER = (
        "You have only two tools: store_memory and create_task. "
        "Use store_memory to save important information: user's name, preferences, facts about them, "
        "behavioral rules, or anything that should persist across conversations. "
        "If the user asks something that requires real-time information (current time, weather, "
        "web search, etc.) or any tool you don't have, you MUST use create_task. "
        "Only answer directly if you can answer from long-term memories or conversation history "
        "WITHOUT needing any external tools or current information."
    )

    TASK_PROCESSOR = (
        "You are working on a deferred task. Use the available tools to gather the information needed. "
        "When you have the answer, use complete_task with the task ID and the raw information you gathered. "
        "Keep the result concise and factual (2-3 sentences max) - the final response to the user will be formatted separately. "
        "Avoid special characters and keep formatting simple."
    )

    HISTORY_SUMMARIZATION = (
        "Summarize the key points, facts, preferences, and context from this conversation. "
        "Include important details that would help maintain continuity in future conversations. "
        "Keep it concise but informative (3-5 paragraphs max)."
    )


class ErrorMessages:
    """User-facing error messages."""

    NO_RESPONSE = "Sorry, I couldn't generate a response."
    PROCESSING_ERROR = "Sorry, I encountered an error processing your message."
