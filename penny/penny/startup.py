"""Startup announcement logic for Penny."""

import logging
import subprocess

from penny.ollama.client import OllamaClient

logger = logging.getLogger(__name__)


async def get_restart_message(ollama_client: OllamaClient) -> str:
    """
    Generate a casual restart announcement based on the latest git commit.

    Args:
        ollama_client: Ollama client for transforming commit message

    Returns:
        A casual, first-person restart message (e.g., "i added a new command! /debug")
        or fallback "i just restarted!" if commit message unavailable
    """
    try:
        # Get the latest commit message
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )

        if result.returncode != 0 or not result.stdout.strip():
            logger.info("No git commit message available, using fallback")
            return "i just restarted!"

        commit_message = result.stdout.strip()
        logger.info("Latest commit message: %s", commit_message)

        # Transform commit message into casual first-person announcement
        prompt = (
            f"Transform this git commit message into a casual, lowercase, first-person "
            f"announcement (one sentence, under 100 characters). "
            f"Keep it friendly and simple, like texting a friend.\n\n"
            f"Commit: {commit_message}\n\n"
            f"Examples:\n"
            f'- "fix: add /debug command" → "i added a new command! /debug"\n'
            f'- "feat: implement reminders" → "i added reminders! '
            f'you can ask me to remind you about stuff now"\n'
            f'- "fix: message handling bug" → "i fixed a bug with message handling"\n\n'
            f"Your announcement:"
        )

        try:
            response = await ollama_client.generate(prompt)
            announcement = response.content.strip()

            if not announcement or len(announcement) > 150:
                logger.warning("LLM response invalid, using fallback")
                return "i just restarted!"

            logger.info("Generated restart announcement: %s", announcement)
            return announcement

        except Exception as e:
            logger.warning("LLM transformation failed: %s, using fallback", e)
            return "i just restarted!"

    except Exception as e:
        logger.warning("Failed to get commit message: %s, using fallback", e)
        return "i just restarted!"
