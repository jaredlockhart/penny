"""MessageAgent for handling incoming user messages."""

from __future__ import annotations

import logging
import re

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse, MessageRole
from penny.ollama.embeddings import deserialize_embedding, find_similar
from penny.prompts import Prompt
from penny.responses import PennyResponse
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
        # Get thread context if quoted
        parent_id = None
        history = None
        if quoted_text:
            parent_id, history = self.db.get_thread_context(quoted_text)

        # Inject user profile context if available
        try:
            user_info = self.db.get_user_info(sender)
            if user_info:
                profile_summary = f"The user's name is {user_info.name}."
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

        prompt_label = (
            "vision" if has_images else ("knowledge" if knowledge_sufficient else "search")
        )
        logger.info("Handling message from %s (prompt=%s)", sender, prompt_label)

        # Run agent (for image messages: disable tools, single pass)
        response = await self.run(
            prompt=content,
            history=history,
            use_tools=not has_images,
            max_steps=1 if has_images else None,
            system_prompt=system_prompt,
        )

        return parent_id, response

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
        if not self._embedding_model_client:
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
            vecs = await self._embedding_model_client.embed(content)
            query_vec = vecs[0]
        except Exception:
            logger.warning("Failed to embed message for entity context, skipping")
            return None, False

        # Find similar entities
        matches = find_similar(
            query_vec,
            candidates,
            top_k=int(self.config.runtime.ENTITY_CONTEXT_TOP_K),
            threshold=self.config.runtime.ENTITY_CONTEXT_THRESHOLD,
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

            fact_texts = [
                f.content for f in facts[: int(self.config.runtime.ENTITY_CONTEXT_MAX_FACTS)]
            ]
            label = f"{entity.name} ({entity.tagline})" if entity.tagline else entity.name
            context_lines.append(f"- {label}: {'; '.join(fact_texts)}")
            total_facts += len(fact_texts)

        if not context_lines:
            return None, False

        context_text = "Relevant knowledge:\n" + "\n".join(context_lines)

        # Knowledge is sufficient when we have enough facts from a high-confidence match
        top_score = matches[0][1]
        is_sufficient = (
            total_facts >= self.config.runtime.KNOWLEDGE_SUFFICIENT_MIN_FACTS
            and top_score >= self.config.runtime.KNOWLEDGE_SUFFICIENT_MIN_SCORE
        )

        logger.info(
            "Entity context: %d entities, %d facts, top_score=%.2f, sufficient=%s",
            len(context_lines),
            total_facts,
            top_score,
            is_sufficient,
        )

        return context_text, is_sufficient
