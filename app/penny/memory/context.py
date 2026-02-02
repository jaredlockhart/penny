"""Context building for conversation history."""


def build_context(history: list, current_message: str) -> str:
    """
    Build conversation context from history.

    Args:
        history: List of Message objects from database
        current_message: The current incoming message

    Returns:
        Formatted context string for Ollama
    """
    context_parts = ["You are Penny, a helpful AI assistant communicating via Signal messages."]

    if history:
        context_parts.append("\nRecent conversation history:")
        for msg in history:
            if msg.direction == "incoming":
                context_parts.append(f"User: {msg.content}")
            else:
                # For outgoing chunks, only include non-chunk messages or first chunk
                if msg.chunk_index is None or msg.chunk_index == 0:
                    context_parts.append(f"Penny: {msg.content}")

    context_parts.append(f"\nUser: {current_message}")
    context_parts.append("Penny:")

    return "\n".join(context_parts)
