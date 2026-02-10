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
        # Check for /test command and validate
        test_mode = False
        actual_content = content
        if content.strip().startswith("/test "):
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

            test_mode = True
            logger.info("Test mode activated for message: %s", actual_content)

        # Determine which database to use
        if test_mode:
            from pathlib import Path

            prod_db_path = Path(self.db.db_path)
            test_db_path = prod_db_path.parent / TEST_DB_PATH.name
            active_db = Database(str(test_db_path))
        else:
            active_db = self.db

        # Execute with the selected database
        parent_id, response = await self._execute_with_db(
            active_db, actual_content, quoted_text, test_mode
        )

        return parent_id, response

    async def _execute_with_db(
        self,
        db: Database,
        content: str,
        quoted_text: str | None,
        test_mode: bool,
    ) -> tuple[int | None, ControllerResponse]:
        """
        Execute the agent with a specific database (DB-agnostic logic).

        Args:
            db: Database instance to use
            content: The message content to process
            quoted_text: Optional quoted text if this is a reply
            test_mode: Whether this is a test mode execution

        Returns:
            Tuple of (parent_id for thread linking, ControllerResponse with answer)
        """
        # Swap database if needed
        original_db = self.db
        if db is not original_db:
            self.db = db
            self._ollama_client.db = db
            for tool in self.tools:
                if hasattr(tool, "db"):
                    tool.db = db  # type: ignore[attr-defined]

        try:
            # Get thread context if quoted
            parent_id = None
            history = None
            if quoted_text:
                parent_id, history = db.get_thread_context(quoted_text)

            # Run agent
            response = await self.run(prompt=content, history=history)

            # Prepend [TEST] to response if in test mode
            if test_mode and response.answer:
                response.answer = f"{TEST_MODE_PREFIX}{response.answer}"

            return parent_id, response
        finally:
            # Restore original database if we swapped
            if db is not original_db:
                self.db = original_db
                self._ollama_client.db = original_db
                for tool in self.tools:
                    if hasattr(tool, "db"):
                        tool.db = original_db  # type: ignore[attr-defined]
