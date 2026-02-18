"""MessageAgent for handling incoming user messages."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse, MessageRole
from penny.config_params import RUNTIME_CONFIG_PARAMS
from penny.constants import PennyConstants
from penny.database.models import ResearchTask
from penny.ollama.embeddings import deserialize_embedding, find_similar
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools.builtin import SearchTool

RESEARCH_MAX_ITERATIONS_DEFAULT = int(
    RUNTIME_CONFIG_PARAMS["RESEARCH_MAX_ITERATIONS"].default_value
)

logger = logging.getLogger(__name__)

GO_KEYWORDS = ("go", "go!", "start", "just go", "go ahead")


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

        # Check if user is replying to a research clarification prompt
        focus_reply = await self._handle_research_focus_reply(sender, content)
        if focus_reply:
            return None, focus_reply

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

        # Retrieve entity knowledge context
        knowledge_sufficient = False
        try:
            entity_context, knowledge_sufficient = await self._retrieve_entity_context(
                content, sender
            )
            if entity_context:
                history = history or []
                history = [*history, (MessageRole.SYSTEM.value, entity_context)]
                logger.debug("Injected entity context (%d chars)", len(entity_context))
        except Exception:
            logger.warning("Entity context retrieval failed, proceeding without")

        # Caption images with vision model, then build combined text prompt
        has_images = bool(images)
        if images:
            captions = [await self.caption_image(img) for img in images]
            caption = ", ".join(captions)
            if content:
                content = PennyResponse.VISION_IMAGE_CONTEXT.format(
                    user_text=content, caption=caption
                )
            else:
                content = PennyResponse.VISION_IMAGE_ONLY_CONTEXT.format(caption=caption)
            logger.info("Built vision prompt: %s", content[:200])

        # Choose system prompt: vision > knowledge-sufficient > default search
        if has_images:
            system_prompt = Prompt.VISION_RESPONSE_PROMPT
        elif knowledge_sufficient:
            system_prompt = Prompt.KNOWLEDGE_PROMPT
        else:
            system_prompt = Prompt.SEARCH_PROMPT

        # Run agent (for image messages: disable tools, single pass)
        response = await self.run(
            prompt=content,
            history=history,
            use_tools=not has_images,
            max_steps=1 if has_images else None,
            system_prompt=system_prompt,
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

            # Get max_iterations from config_params default
            max_iterations = RESEARCH_MAX_ITERATIONS_DEFAULT

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
            answer=PennyResponse.RESEARCH_CONTINUATION.format(content=content)
        )

    async def _handle_research_focus_reply(
        self, sender: str, content: str
    ) -> ControllerResponse | None:
        """
        Check if sender has an awaiting_focus research task and handle focus reply.

        If the user says "go" (or similar), start research with no focus.
        Otherwise, use LLM to interpret the user's reply against the presented
        options and extract a clear focus description for the research report.

        Returns:
            ControllerResponse if this is a focus reply, None otherwise
        """
        try:
            with Session(self.db.engine) as session:
                task = session.exec(
                    select(ResearchTask).where(
                        ResearchTask.thread_id == sender,
                        ResearchTask.status == "awaiting_focus",
                    )
                ).first()

                if not task:
                    return None

                # "go" or similar → start with no focus
                if content.strip().lower() in GO_KEYWORDS:
                    task.status = "in_progress"
                    session.add(task)
                    session.commit()
                    logger.info("Research task %d started without focus (user said 'go')", task.id)
                    return ControllerResponse(
                        answer=PennyResponse.RESEARCH_STARTED.format(topic=task.topic)
                    )

                # Use LLM to interpret user's reply against the options
                focus = await self._extract_focus(task.options, content.strip())

                task.focus = focus
                task.status = "in_progress"
                session.add(task)
                session.commit()
                logger.info("Research task %d started with focus: %s", task.id, task.focus)
                return ControllerResponse(
                    answer=PennyResponse.RESEARCH_STARTED_WITH_FOCUS.format(
                        topic=task.topic, focus=task.focus
                    )
                )
        except Exception:
            # Table may not exist (e.g., in test mode databases)
            return None

    async def _extract_focus(self, options: str | None, user_reply: str) -> str:
        """Use LLM to interpret the user's reply and extract a report focus."""
        if not options:
            # No stored options (e.g., old task) — use reply directly
            return user_reply

        user_content = f"Options presented:\n{options}\n\nUser replied: {user_reply}"

        response = await self._ollama_client.chat(
            messages=[
                {"role": "system", "content": Prompt.RESEARCH_FOCUS_EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        extracted = response.content.strip()
        # Fall back to raw reply if LLM returns empty
        return extracted or user_reply

    async def _retrieve_entity_context(self, content: str, sender: str) -> tuple[str | None, bool]:
        """
        Retrieve relevant entity knowledge for a user's message.

        Embeds the message, finds similar entities via cosine similarity,
        and formats their facts as context text.

        Args:
            content: The user's message text
            sender: The sender identifier

        Returns:
            Tuple of (context_text, is_sufficient) where:
            - context_text: Formatted entity facts, or None if no matches
            - is_sufficient: True if enough knowledge exists to potentially
              answer without search
        """
        if not self.embedding_model:
            return None, False

        # Load user entities with embeddings
        entities = self.db.get_user_entities(sender)
        candidates: list[tuple[int, list[float]]] = []
        for entity in entities:
            if entity.embedding is None or entity.id is None:
                continue
            candidates.append((entity.id, deserialize_embedding(entity.embedding)))

        if not candidates:
            return None, False

        # Embed the user's message
        try:
            vecs = await self._ollama_client.embed(content, model=self.embedding_model)
            query_vec = vecs[0]
        except Exception:
            logger.warning("Failed to embed message for entity context, skipping")
            return None, False

        # Find similar entities
        matches = find_similar(
            query_vec,
            candidates,
            top_k=PennyConstants.ENTITY_CONTEXT_TOP_K,
            threshold=PennyConstants.ENTITY_CONTEXT_THRESHOLD,
        )

        if not matches:
            return None, False

        # Build context from matched entities
        entity_map = {e.id: e for e in entities if e.id is not None}
        context_lines: list[str] = []
        total_facts = 0

        for entity_id, _score in matches:
            entity = entity_map.get(entity_id)
            if not entity or entity.id is None:
                continue

            facts = self.db.get_entity_facts(entity.id)
            if not facts:
                continue

            fact_texts = [f.content for f in facts[: PennyConstants.ENTITY_CONTEXT_MAX_FACTS]]
            context_lines.append(f"- {entity.name}: {'; '.join(fact_texts)}")
            total_facts += len(fact_texts)

        if not context_lines:
            return None, False

        context_text = "Relevant knowledge:\n" + "\n".join(context_lines)

        # Knowledge is sufficient when we have enough facts from a high-confidence match
        top_score = matches[0][1]
        is_sufficient = (
            total_facts >= PennyConstants.KNOWLEDGE_SUFFICIENT_MIN_FACTS
            and top_score >= PennyConstants.KNOWLEDGE_SUFFICIENT_MIN_SCORE
        )

        logger.info(
            "Entity context: %d entities, %d facts, top_score=%.2f, sufficient=%s",
            len(context_lines),
            total_facts,
            top_score,
            is_sufficient,
        )

        return context_text, is_sufficient
