"""Constants for Penny agent."""


class SystemPrompts:
    """System prompts for different contexts."""

    MESSAGE_HANDLER = (
        "You have only two tools: store_memory and create_task. "
        "If the user asks something that requires real-time information (current time, weather, "
        "web search, etc.) or any tool you don't have, you MUST use create_task. "
        "Only answer directly if you can answer from long-term memories or conversation history "
        "WITHOUT needing any external tools or current information."
    )

    TASK_PROCESSOR = (
        "You have pending tasks. Use list_tasks to see them, "
        "then work on them using available tools. "
        "When complete, use complete_task with the final answer."
    )


class ErrorMessages:
    """User-facing error messages."""

    NO_RESPONSE = "Sorry, I couldn't generate a response."
    PROCESSING_ERROR = "Sorry, I encountered an error processing your message."
