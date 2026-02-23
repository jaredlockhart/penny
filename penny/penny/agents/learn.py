"""Scheduled worker for /learn command research — one search step per tick."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.agents.models import GeneratedQuery
from penny.constants import PennyConstants
from penny.prompts import Prompt
from penny.tools.models import SearchResult

if TYPE_CHECKING:
    from penny.tools import Tool

logger = logging.getLogger(__name__)


class LearnAgent(Agent):
    """Background worker that processes /learn prompts one search step at a time.

    Each execute() call:
    1. Finds the next active LearnPrompt (oldest first, across all users)
    2. Generates a query (initial or followup based on existing search logs)
    3. Executes one search
    4. Decrements searches_remaining
    5. If searches_remaining == 0 or LLM returns empty query, marks completed

    Returns True if work was done, False if no pending work.
    """

    def __init__(self, search_tool: Tool | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._search_tool = search_tool

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "learn"

    async def execute(self) -> bool:
        """Run one search step for the next pending learn prompt.

        Gated by unextracted learn search logs: if any previous learn search
        hasn't been extracted yet, skip this tick to let ExtractionPipeline
        catch up. This ensures topics flow through the pipeline one step at a
        time (search → extract → notify) rather than all searches completing
        before extraction starts.
        """
        if not self._search_tool:
            return False

        # Gate: wait for all previous learn searches to be extracted
        if self.db.has_unextracted_learn_search_logs():
            return False

        learn_prompt = self.db.get_next_active_learn_prompt()
        if learn_prompt is None:
            return False

        assert learn_prompt.id is not None
        topic = learn_prompt.prompt_text

        # Reconstruct previous search results from DB
        search_logs = self.db.get_search_logs_by_learn_prompt(learn_prompt.id)
        previous_results = [sl.response for sl in search_logs if sl.response]

        # Generate query: initial if no previous searches, followup otherwise
        if not previous_results:
            query = await self._generate_initial_query(topic)
        else:
            query = await self._generate_followup_query(topic, previous_results)
            if query is None:
                # LLM decided research is complete
                self.db.update_learn_prompt_status(
                    learn_prompt.id, PennyConstants.LearnPromptStatus.COMPLETED
                )
                logger.info(
                    "Learn research completed (LLM done) for '%s': %d searches",
                    topic,
                    len(search_logs),
                )
                return True

        # Execute search
        await self._search(query, learn_prompt.id)
        self.db.decrement_learn_prompt_searches(learn_prompt.id)

        # Check if this was the last search
        refreshed = self.db.get_learn_prompt(learn_prompt.id)
        if refreshed and refreshed.searches_remaining <= 0:
            self.db.update_learn_prompt_status(
                learn_prompt.id, PennyConstants.LearnPromptStatus.COMPLETED
            )
            search_count = len(search_logs) + 1
            logger.info("Learn research completed for '%s': %d searches", topic, search_count)

        return True

    async def _generate_initial_query(self, topic: str) -> str:
        """Generate the first search query for a topic via LLM."""
        prompt = f"{Prompt.LEARN_INITIAL_QUERY_PROMPT}\n\nTopic: {topic}"
        try:
            response = await self._foreground_model_client.generate(
                prompt=prompt,
                tools=None,
                format=GeneratedQuery.model_json_schema(),
            )
            result = GeneratedQuery.model_validate_json(response.content)
            if result.query.strip():
                return result.query.strip()
        except Exception as e:
            logger.error("Failed to generate initial query for '%s': %s", topic, e)
        # Fallback: use the topic as-is
        return topic

    async def _generate_followup_query(self, topic: str, previous_results: list[str]) -> str | None:
        """Generate the next search query based on previous results.

        Returns the query string, or None if research is complete.
        """
        results_text = "\n\n---\n\n".join(
            f"Search {i + 1}:\n{text[:1000]}" for i, text in enumerate(previous_results)
        )
        prompt = Prompt.LEARN_FOLLOWUP_QUERY_PROMPT.format(
            topic=topic, previous_results=results_text
        )
        try:
            response = await self._foreground_model_client.generate(
                prompt=prompt,
                tools=None,
                format=GeneratedQuery.model_json_schema(),
            )
            result = GeneratedQuery.model_validate_json(response.content)
            query = result.query.strip()
            return query if query else None
        except Exception as e:
            logger.error("Failed to generate followup query for '%s': %s", topic, e)
            return None

    async def _search(self, query: str, learn_prompt_id: int) -> str | None:
        """Execute search via SearchTool with provenance. Returns text or None."""
        assert self._search_tool is not None
        try:
            result = await self._search_tool.execute(
                query=query,
                skip_images=True,
                trigger=PennyConstants.SearchTrigger.LEARN_COMMAND,
                learn_prompt_id=learn_prompt_id,
            )
            if isinstance(result, SearchResult):
                return result.text
            return str(result) if result else None
        except Exception as e:
            logger.error("Learn research search failed: %s", e)
            return None
