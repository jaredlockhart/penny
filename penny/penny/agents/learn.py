"""Scheduled worker for /learn command research — one search step per tick."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.agents.base import Agent
from penny.agents.models import GeneratedQuery
from penny.constants import PennyConstants
from penny.database.models import LearnPrompt
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
        learn_prompt = self._check_preconditions()
        if learn_prompt is None:
            return False

        assert learn_prompt.id is not None
        completed_early = await self._execute_search_step(learn_prompt)
        if not completed_early:
            self._check_completion(learn_prompt)
        return True

    def _check_preconditions(self) -> LearnPrompt | None:
        """Check if we should run: tool available, no unextracted logs, active prompt.

        Returns the next active LearnPrompt, or None to skip this tick.
        """
        if not self._search_tool:
            return None

        if self.db.searches.has_unextracted_learn_logs():
            return None

        return self.db.learn_prompts.get_next_active()

    async def _execute_search_step(self, learn_prompt: LearnPrompt) -> bool:
        """Generate a query and execute one search. Returns True if LLM ended research early."""
        assert learn_prompt.id is not None
        topic = learn_prompt.prompt_text

        search_logs = self.db.searches.get_by_learn_prompt(learn_prompt.id)
        previous_results = [sl.response for sl in search_logs if sl.response]

        if not previous_results:
            query = await self._generate_initial_query(topic)
        else:
            query = await self._generate_followup_query(topic, previous_results)
            if query is None:
                self.db.learn_prompts.update_status(
                    learn_prompt.id, PennyConstants.LearnPromptStatus.COMPLETED
                )
                logger.info(
                    "Learn research completed (LLM done) for '%s': %d searches",
                    topic,
                    len(search_logs),
                )
                return True

        await self._search(query, learn_prompt.id)
        self.db.learn_prompts.decrement_searches(learn_prompt.id)
        return False

    def _check_completion(self, learn_prompt: LearnPrompt) -> None:
        """Mark the learn prompt as completed if no searches remain."""
        assert learn_prompt.id is not None
        topic = learn_prompt.prompt_text

        refreshed = self.db.learn_prompts.get(learn_prompt.id)
        if refreshed and refreshed.searches_remaining <= 0:
            self.db.learn_prompts.update_status(
                learn_prompt.id, PennyConstants.LearnPromptStatus.COMPLETED
            )
            search_logs = self.db.searches.get_by_learn_prompt(learn_prompt.id)
            logger.info("Learn research completed for '%s': %d searches", topic, len(search_logs))

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
