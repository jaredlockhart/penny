"""Built-in tools."""

from datetime import datetime
from typing import Any

from perplexity import Perplexity

from penny.tools.base import Tool


class GetCurrentTimeTool(Tool):
    """Get the current date and time."""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "Get the current date and time in ISO format. Use this when the user asks about the current time or date."

    @property
    def parameters(self) -> dict[str, Any]:
        # No parameters needed
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs) -> str:
        """Get current time."""
        return datetime.now().isoformat()


class StoreMemoryTool(Tool):
    """Tool for storing long-term memories."""

    def __init__(self, db):
        """
        Initialize the tool with database access.

        Args:
            db: Database instance for storing memories
        """
        self.db = db

    @property
    def name(self) -> str:
        return "store_memory"

    @property
    def description(self) -> str:
        return (
            "Store a long-term memory, fact, preference, or rule that should be remembered "
            "across all conversations. Use this for: user names, preferences, behavioral rules, "
            "or any important information that should persist. Examples: 'My name is Jared', "
            "'Always speak in lowercase', 'User prefers concise answers'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "The fact, preference, or rule to remember",
                }
            },
            "required": ["memory"],
        }

    async def execute(self, memory: str, **kwargs) -> str:
        """
        Store a memory in the database.

        Args:
            memory: The memory content to store

        Returns:
            Confirmation message
        """
        self.db.store_memory(memory)
        return f"Stored memory: {memory}"


class PerplexitySearchTool(Tool):
    """Tool for searching the web using Perplexity AI."""

    def __init__(self, api_key: str):
        """
        Initialize the tool with Perplexity API key.

        Args:
            api_key: Perplexity API key
        """
        self.client = Perplexity(api_key=api_key)

    @property
    def name(self) -> str:
        return "perplexity_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information using Perplexity AI. "
            "Use this when you need up-to-date information, facts, news, or "
            "answers to questions that require real-time data or information "
            "beyond your training data."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to ask Perplexity",
                }
            },
            "required": ["query"],
        }

    async def execute(self, query: str, **kwargs) -> str:
        """
        Execute a search using Perplexity.

        Args:
            query: The search query

        Returns:
            Search results as a string
        """
        try:
            response = self.client.responses.create(
                preset="pro-search",
                input=query,
            )
            if response.output_text:
                return response.output_text
            return "No results found"
        except Exception as e:
            return f"Error performing search: {str(e)}"
