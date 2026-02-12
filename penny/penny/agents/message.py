"""MessageAgent for handling incoming user messages."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse, MessageRole
from penny.constants import VISION_IMAGE_CONTEXT, VISION_IMAGE_ONLY_CONTEXT
from penny.database.models import ResearchTask
from penny.tools.builtin import SearchTool

logger = logging.getLogger(__name__)


class MessageAgent(Agent):
    """Agent for handling incoming user messages."""

    async def handle(
        self,
        content: str,
        sender: str,
        quoted_text: str | None = None,
        images: list[str] | None = None,
    ) -> tuple[int | None, ControllerResponse]:
        """
        Handle an incoming message by preparing context and running the agent.

        Args:
            content: The message content from the user
            sender: The sender identifier
            quoted_text: Optional quoted text if this is a reply

        Returns:
            Tuple of (parent_id for thread linking, ControllerResponse with answer)
        """
        # Check if this is a quote-reply to a research report
        if quoted_text:
            research_continuation = self._handle_research_continuation(sender, content, quoted_text)
            if research_continuation:
                return None, research_continuation

        # Get thread context if quoted
        parent_id = None
        history = None
        if quoted_text:
            parent_id, history = self.db.get_thread_context(quoted_text)

        # Inject user profile context if available
        try:
            user_info = self.db.get_user_info(sender)
            if user_info:
                profile_summary = (
                    f"User context: {user_info.name}, {user_info.location} ({user_info.timezone})"
                )
                # Prepend profile context to history
                history = history or []
                history = [(MessageRole.SYSTEM.value, profile_summary), *history]
                logger.debug("Injected profile context for %s", sender)

                # Redact user name from outbound search queries, but only
                # if the user didn't use their own name in the message
                name = user_info.name
                user_said_name = bool(re.search(rf"\b{re.escape(name)}\b", content, re.IGNORECASE))
                search_tool = self._tool_registry.get("search")
                if isinstance(search_tool, SearchTool):
                    search_tool.redact_terms = [] if user_said_name else [name]
        except Exception:
            # Silently skip if userinfo table doesn't exist (e.g., in test mode)
            pass

        # Caption images with vision model, then build combined text prompt
        has_images = bool(images)
        if images:
            captions = [await self.caption_image(img) for img in images]
            caption = ", ".join(captions)
            if content:
                content = VISION_IMAGE_CONTEXT.format(user_text=content, caption=caption)
            else:
                content = VISION_IMAGE_ONLY_CONTEXT.format(caption=caption)
            logger.info("Built vision prompt: %s", content[:200])

        # Run agent (for image messages: disable tools and use single pass)
        response = await self.run(
            prompt=content,
            history=history,
            use_tools=not has_images,
            max_steps=1 if has_images else None,
        )

        return parent_id, response

    def _handle_research_continuation(
        self, sender: str, content: str, quoted_text: str
    ) -> ControllerResponse | None:
        """
        Check if quoted_text is a research report and create continuation task.

        Args:
            sender: User identifier
            content: User's new research prompt
            quoted_text: The quoted message text

        Returns:
            ControllerResponse if this is a research continuation, None otherwise
        """
        # Find the quoted message in database
        quoted_message = self.db.find_outgoing_by_content(quoted_text)
        if not quoted_message or not quoted_message.id:
            return None

        # Check if this message is associated with a completed research task
        with Session(self.db.engine) as session:
            research_task = session.exec(
                select(ResearchTask).where(ResearchTask.message_id == str(quoted_message.id))
            ).first()

            if not research_task or research_task.status != "completed":
                return None

            logger.info("Detected research continuation for task %d: %s", research_task.id, content)

            # Get max_iterations from config (not from database, use default)
            # We can't easily access config here, so we'll use the default value
            max_iterations = 10  # Default from config_params.py

            # Create new research task as continuation
            new_task = ResearchTask(
                thread_id=research_task.thread_id,
                parent_task_id=research_task.id,
                topic=content,  # User's refinement becomes the new topic
                status="in_progress",
                max_iterations=max_iterations,
                created_at=datetime.now(UTC),
            )
            session.add(new_task)
            session.commit()
            logger.info("Created continuation research task %d", new_task.id)

        return ControllerResponse(
            answer=f"Ok, continuing research with focus: {content}. I'll post results when done."
        )
