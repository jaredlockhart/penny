"""MessageAgent for handling incoming user messages."""

import logging

from penny.agent.base import Agent
from penny.agent.models import ControllerResponse
from penny.constants import TEST_DB_PATH, TEST_MODE_PREFIX
from penny.database import Database

logger = logging.getLogger(__name__)


class MessageAgent(Agent):
    """Agent for handling incoming user messages."""

    async def handle(
        self,
        content: str,
        sender: str,
        quoted_text: str | None = None,
    ) -> tuple[int | None, ControllerResponse]:
        """
        Handle an incoming message by preparing context and running the agent.

        Args:
            content: The message content from the user
            sender: The sender identifier (unused in this version)
            quoted_text: Optional quoted text if this is a reply

        Returns:
            Tuple of (parent_id for thread linking, ControllerResponse with answer)
        """
        # Check for /test command
        test_mode = False
        actual_content = content
        if content.strip().startswith("/test "):
            test_mode = True
            # Strip /test prefix
            actual_content = content.strip()[6:]  # len("/test ")

            # Reject nested commands
            if actual_content.strip().startswith("/"):
                error_response = ControllerResponse(
                    answer="Nested commands are not supported in test mode."
                )
                return None, error_response

            # Reject threaded test prompts
            if quoted_text:
                error_response = ControllerResponse(answer="Test prompts cannot be threaded.")
                return None, error_response

            logger.info("Test mode activated for message: %s", actual_content)

        parent_id = None
        history = None
        if quoted_text:
            parent_id, history = self.db.get_thread_context(quoted_text)

        # Run agent with test database if in test mode
        if test_mode:
            # Create test database path in same directory as production db
            from pathlib import Path

            prod_db_path = Path(self.db.db_path)
            test_db_path = prod_db_path.parent / TEST_DB_PATH.name
            # Create a temporary test database instance
            test_db = Database(str(test_db_path))
            # Swap database for this execution
            original_db = self.db
            self.db = test_db
            self._ollama_client.db = test_db
            for tool in self.tools:
                if hasattr(tool, "db"):
                    tool.db = test_db  # type: ignore[attr-defined]

            try:
                response = await self.run(prompt=actual_content, history=history)
                # Prepend [TEST] to response
                if response.answer:
                    response.answer = f"{TEST_MODE_PREFIX}{response.answer}"
            finally:
                # Restore original database
                self.db = original_db
                self._ollama_client.db = original_db
                for tool in self.tools:
                    if hasattr(tool, "db"):
                        tool.db = original_db  # type: ignore[attr-defined]
        else:
            response = await self.run(prompt=actual_content, history=history)

        return parent_id, response
