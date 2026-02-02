"""Built-in tools."""

from datetime import datetime

from perplexity import Perplexity

from penny.tools.base import Tool


class GetCurrentTimeTool(Tool):
    """Get the current date and time."""

    name = "get_current_time"
    description = "Get the current date and time in ISO format. Use this when the user asks about the current time or date."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        """Get current time."""
        return datetime.now().isoformat()


class StoreMemoryTool(Tool):
    """Tool for storing long-term memories."""

    name = "store_memory"
    description = (
        "Store a long-term memory, fact, preference, or rule that should be remembered "
        "across all conversations. Use this for: user names, preferences, behavioral rules, "
        "or any important information that should persist. Examples: 'My name is Jared', "
        "'Always speak in lowercase', 'User prefers concise answers'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {
                "type": "string",
                "description": "The fact, preference, or rule to remember",
            }
        },
        "required": ["memory"],
    }

    def __init__(self, db):
        """
        Initialize the tool with database access.

        Args:
            db: Database instance for storing memories
        """
        self.db = db

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

    name = "perplexity_search"
    description = (
        "Search the web for current information using Perplexity AI. "
        "Use this when you need up-to-date information, facts, news, or "
        "answers to questions that require real-time data or information "
        "beyond your training data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or question to ask Perplexity",
            }
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str):
        """
        Initialize the tool with Perplexity API key.

        Args:
            api_key: Perplexity API key
        """
        self.client = Perplexity(api_key=api_key)

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


class CreateTaskTool(Tool):
    """Tool for creating deferred tasks."""

    name = "create_task"
    description = (
        "Create a task to work on later. Use this when you need to use other tools "
        "(like search, time, memory) but want to defer the work. After creating a task, "
        "you should respond to the user acknowledging that you'll work on it. "
        "The task will be processed in the background."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "What needs to be done in this task",
            },
            "acknowledgment": {
                "type": "string",
                "description": "Brief casual message to send to user (5-10 words, lowercase)",
            }
        },
        "required": ["description", "acknowledgment"],
    }

    def __init__(self, db, agent):
        """
        Initialize with database and agent reference.

        Args:
            db: Database instance
            agent: PennyAgent instance (for accessing user_number)
        """
        self.db = db
        self.agent = agent

    async def execute(self, description: str, acknowledgment: str, **kwargs) -> str:
        """
        Create a task.

        Args:
            description: Task description
            acknowledgment: Message to acknowledge task creation

        Returns:
            Confirmation with acknowledgment to send
        """
        task = self.db.create_task(description, self.agent.user_number)
        return f"Task {task.id} created. Send this to user: {acknowledgment}"


class ListTasksTool(Tool):
    """Tool for listing pending tasks."""

    name = "list_tasks"
    description = (
        "List all pending tasks that need to be worked on. "
        "Use this when idle to see if there's work to do."
    )
    parameters = {"type": "object", "properties": {}}

    def __init__(self, db):
        """
        Initialize with database.

        Args:
            db: Database instance
        """
        self.db = db

    async def execute(self, **kwargs) -> str:
        """
        List pending tasks.

        Returns:
            List of pending tasks with IDs and descriptions
        """
        tasks = self.db.get_pending_tasks()
        if not tasks:
            return "No pending tasks"

        result = "Pending tasks:\n"
        for task in tasks:
            result += f"- Task {task.id}: {task.content} (from {task.requester})\n"
        return result


class CompleteTaskTool(Tool):
    """Tool for marking tasks as complete."""

    name = "complete_task"
    description = (
        "Mark a task as completed with the final result. "
        "The result will be sent to the user who requested the task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "integer",
                "description": "ID of the task to complete",
            },
            "result": {
                "type": "string",
                "description": "Final answer/result to send to user",
            }
        },
        "required": ["task_id", "result"],
    }

    def __init__(self, db):
        """
        Initialize with database.

        Args:
            db: Database instance
        """
        self.db = db

    async def execute(self, task_id: int, result: str, **kwargs) -> str:
        """
        Complete a task.

        Args:
            task_id: Task ID
            result: Final result

        Returns:
            Confirmation
        """
        self.db.complete_task(task_id, result)
        return f"Task {task_id} completed. Result will be sent to user: {result}"
