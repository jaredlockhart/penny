"""MessageAgent for handling incoming user messages."""

from __future__ import annotations

import logging
import re

from penny.agents.base import Agent
from penny.agents.models import ControllerResponse, MessageRole
from penny.database.models import Entity
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
        parent_id, history = self._resolve_thread_context(quoted_text)
        history = self._inject_profile_context(sender, history, content)
        history, knowledge_sufficient = await self._inject_entity_context(content, sender, history)
        content, has_images = await self._process_images(content, images)
        system_prompt = self._select_system_prompt(has_images, knowledge_sufficient)

        prompt_label = (
            "vision" if has_images else ("knowledge" if knowledge_sufficient else "search")
        )
        logger.info("Handling message from %s (prompt=%s)", sender, prompt_label)

        response = await self.run(
            prompt=content,
            history=history,
            use_tools=not has_images,
            max_steps=1 if has_images else None,
            system_prompt=system_prompt,
        )

        return parent_id, response

    def _resolve_thread_context(
        self, quoted_text: str | None
    ) -> tuple[int | None, list[tuple[str, str]] | None]:
        """Resolve thread context from a quoted message.

        Returns (parent_id, history) — both None if no quoted text.
        """
        if not quoted_text:
            return None, None
        return self.db.messages.get_thread_context(quoted_text)

    def _inject_profile_context(
        self,
        sender: str,
        history: list[tuple[str, str]] | None,
        content: str,
    ) -> list[tuple[str, str]] | None:
        """Prepend user profile context to history and configure search redaction.

        Returns the updated history (unchanged if no profile is available).
        """
        try:
            user_info = self.db.users.get_info(sender)
            if not user_info:
                return history

            profile_summary = f"The user's name is {user_info.name}."
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

        return history

    async def _inject_entity_context(
        self,
        content: str,
        sender: str,
        history: list[tuple[str, str]] | None,
    ) -> tuple[list[tuple[str, str]] | None, bool]:
        """Retrieve entity knowledge and append it to history.

        Returns (updated_history, knowledge_sufficient).
        """
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

        return history, knowledge_sufficient

    async def _process_images(self, content: str, images: list[str] | None) -> tuple[str, bool]:
        """Caption images with vision model and build combined text prompt.

        Returns (updated_content, has_images).
        """
        if not images:
            return content, False

        captions = [await self.caption_image(img) for img in images]
        caption = ", ".join(captions)
        if content:
            content = PennyResponse.VISION_IMAGE_CONTEXT.format(user_text=content, caption=caption)
        else:
            content = PennyResponse.VISION_IMAGE_ONLY_CONTEXT.format(caption=caption)
        logger.info("Built vision prompt: %s", content[:200])

        return content, True

    def _select_system_prompt(self, has_images: bool, knowledge_sufficient: bool) -> str:
        """Choose system prompt: vision > knowledge-sufficient > default search."""
        if has_images:
            return Prompt.VISION_RESPONSE_PROMPT
        if knowledge_sufficient:
            return Prompt.KNOWLEDGE_PROMPT
        return Prompt.SEARCH_PROMPT

    # ── Entity context retrieval ──────────────────────────────────────────────

    async def _retrieve_entity_context(self, content: str, sender: str) -> tuple[str | None, bool]:
        """
        Retrieve relevant entity knowledge for a user's message.

        Embeds the message, finds similar entities via cosine similarity,
        and formats their facts as context text.

        Returns:
            Tuple of (context_text, is_sufficient) where:
            - context_text: Formatted entity facts, or None if no matches
            - is_sufficient: True if enough knowledge exists to potentially
              answer without search
        """
        if not self._embedding_model_client:
            return None, False

        candidates, entities = self._load_entity_candidates(sender)
        if not candidates:
            return None, False

        query_vec = await self._embed_message(content)
        if query_vec is None:
            return None, False

        matches = find_similar(
            query_vec,
            candidates,
            top_k=int(self.config.runtime.ENTITY_CONTEXT_TOP_K),
            threshold=self.config.runtime.ENTITY_CONTEXT_THRESHOLD,
        )
        if not matches:
            return None, False

        context_text, total_facts = self._build_entity_context_text(matches, entities)
        if context_text is None:
            return None, False

        is_sufficient = self._assess_knowledge_sufficiency(total_facts, matches[0][1])

        logger.info(
            "Entity context: %d matches, %d facts, top_score=%.2f, sufficient=%s",
            len(matches),
            total_facts,
            matches[0][1],
            is_sufficient,
        )

        return context_text, is_sufficient

    def _load_entity_candidates(
        self, sender: str
    ) -> tuple[list[tuple[int, list[float]]], list[Entity]]:
        """Load user entities with embeddings as (id, vector) pairs.

        Returns (candidates, entities) — candidates may be empty.
        """
        entities = self.db.entities.get_for_user(sender)
        candidates: list[tuple[int, list[float]]] = []
        for entity in entities:
            if entity.embedding is None or entity.id is None:
                continue
            candidates.append((entity.id, deserialize_embedding(entity.embedding)))
        return candidates, entities

    async def _embed_message(self, content: str) -> list[float] | None:
        """Embed the user's message. Returns None on failure."""
        if not self._embedding_model_client:
            return None
        try:
            vecs = await self._embedding_model_client.embed(content)
            return vecs[0]
        except Exception:
            logger.warning("Failed to embed message for entity context, skipping")
            return None

    def _build_entity_context_text(
        self,
        matches: list[tuple[int, float]],
        entities: list[Entity],
    ) -> tuple[str | None, int]:
        """Build formatted context text from matched entities and their facts.

        Returns (context_text, total_facts) — context_text is None if no facts found.
        """
        entity_map = {e.id: e for e in entities if e.id is not None}
        context_lines: list[str] = []
        total_facts = 0

        for entity_id, _score in matches:
            entity = entity_map.get(entity_id)
            if not entity or entity.id is None:
                continue

            facts = self.db.facts.get_for_entity(entity.id)
            if not facts:
                continue

            fact_texts = [
                f.content for f in facts[: int(self.config.runtime.ENTITY_CONTEXT_MAX_FACTS)]
            ]
            label = f"{entity.name} ({entity.tagline})" if entity.tagline else entity.name
            context_lines.append(f"- {label}: {'; '.join(fact_texts)}")
            total_facts += len(fact_texts)

        if not context_lines:
            return None, 0

        context_text = "Relevant knowledge:\n" + "\n".join(context_lines)
        return context_text, total_facts

    def _assess_knowledge_sufficiency(self, total_facts: int, top_score: float) -> bool:
        """Determine if retrieved knowledge is sufficient to skip search."""
        return (
            total_facts >= self.config.runtime.KNOWLEDGE_SUFFICIENT_MIN_FACTS
            and top_score >= self.config.runtime.KNOWLEDGE_SUFFICIENT_MIN_SCORE
        )
