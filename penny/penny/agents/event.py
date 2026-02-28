"""EventAgent — polls news for followed topics, creates events, links entities."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import Event, FollowPrompt
from penny.ollama.embeddings import serialize_embedding
from penny.ollama.similarity import (
    DedupStrategy,
    check_relevance,
    embed_text,
    is_embedding_duplicate,
)
from penny.prompts import Prompt
from penny.tools.news import NewsArticle, NewsTool

logger = logging.getLogger(__name__)

_HEADLINE_STRIP_RE = re.compile(r"[^a-z0-9\s]")


class ExtractedEntities(BaseModel):
    """LLM-extracted entity names from a news article."""

    entities: list[str] = Field(default_factory=list)


class EventAgent(Agent):
    """Background agent that polls news for followed topics.

    Each execute() call:
    1. Checks preconditions (news tool, poll interval)
    2. Gets next FollowPrompt to poll (round-robin)
    3. Fetches articles from NewsAPI
    4. Filters by relevance (embedding similarity to topic)
    5. Deduplicates (URL → headline → embedding similarity)
    6. Creates Event records for new articles
    7. Links events to entities (full mode — creates new entities)
    8. Updates last_polled_at
    """

    def __init__(self, news_tool: NewsTool | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._news_tool = news_tool

    @property
    def name(self) -> str:
        """Task name for logging."""
        return "event"

    async def execute(self) -> bool:
        """Poll news for the next follow prompt. Returns True if work was done."""
        follow_prompt = self._check_preconditions()
        if follow_prompt is None:
            return False

        assert follow_prompt.id is not None
        articles = await self._fetch_articles(follow_prompt)
        articles = await self._filter_relevant(articles, follow_prompt)
        new_articles = await self._deduplicate(articles, follow_prompt)
        events_created = await self._create_events(new_articles, follow_prompt)
        self.db.follow_prompts.update_last_polled(follow_prompt.id)

        if events_created:
            logger.info(
                "EventAgent created %d events for '%s'",
                len(events_created),
                follow_prompt.prompt_text,
            )
        return len(events_created) > 0

    # --- Preconditions ---

    def _check_preconditions(self) -> FollowPrompt | None:
        """Check tool availability and find the next due prompt. Returns prompt or None."""
        if not self._news_tool:
            return None

        for prompt in self.db.follow_prompts.get_active_by_poll_priority():
            if self._poll_interval_elapsed(prompt):
                return prompt
        return None

    def _poll_interval_elapsed(self, prompt: FollowPrompt) -> bool:
        """Check if the prompt's cadence interval has elapsed since last poll."""
        if prompt.last_polled_at is None:
            return True
        interval = PennyConstants.FOLLOW_CADENCE_SECONDS.get(
            prompt.cadence,
            PennyConstants.FOLLOW_CADENCE_SECONDS[PennyConstants.FOLLOW_DEFAULT_CADENCE],
        )
        last = prompt.last_polled_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        elapsed = (datetime.now(UTC) - last).total_seconds()
        return elapsed >= interval

    # --- Fetch ---

    async def _fetch_articles(self, follow_prompt: FollowPrompt) -> list[NewsArticle]:
        """Query NewsAPI with the follow prompt's query terms."""
        assert self._news_tool is not None
        try:
            query_terms = json.loads(follow_prompt.query_terms)
        except json.JSONDecodeError, TypeError:
            query_terms = [follow_prompt.prompt_text]

        window_days = int(self.config.runtime.EVENT_DEDUP_WINDOW_DAYS)
        from_date = datetime.now(UTC) - timedelta(days=window_days)
        return await self._news_tool.search(query_terms, from_date=from_date)

    # --- Relevance ---

    async def _filter_relevant(
        self, articles: list[NewsArticle], follow_prompt: FollowPrompt
    ) -> list[NewsArticle]:
        """Filter articles by embedding similarity to the follow prompt topic."""
        topic_vec = await embed_text(self._embedding_model_client, follow_prompt.prompt_text)
        if topic_vec is None:
            return articles  # No embedding model — pass all through

        threshold = self.config.runtime.EVENT_RELEVANCE_THRESHOLD
        relevant: list[NewsArticle] = []
        for article in articles:
            article_vec = await embed_text(self._embedding_model_client, article.title)
            if (
                article_vec is None
                or check_relevance(article_vec, topic_vec, threshold) is not None
            ):
                relevant.append(article)
            else:
                logger.debug("Relevance: rejected '%s' (below %.2f)", article.title[:60], threshold)

        logger.debug(
            "Relevance: %d articles → %d relevant for '%s'",
            len(articles),
            len(relevant),
            follow_prompt.prompt_text,
        )
        return relevant

    # --- Dedup ---

    async def _deduplicate(
        self, articles: list[NewsArticle], follow_prompt: FollowPrompt
    ) -> list[NewsArticle]:
        """Three-layer dedup: URL → normalized headline → semantic (TCR OR embedding)."""
        window_days = int(self.config.runtime.EVENT_DEDUP_WINDOW_DAYS)
        recent_events = self.db.events.get_recent(follow_prompt.user, days=window_days)
        new_articles: list[NewsArticle] = []

        for article in articles:
            if self._is_url_duplicate(article, recent_events):
                continue
            if self._is_headline_duplicate(article, recent_events):
                continue
            if await self._is_semantic_duplicate(article, recent_events):
                continue
            new_articles.append(article)

        logger.debug(
            "Dedup: %d articles → %d new for '%s'",
            len(articles),
            len(new_articles),
            follow_prompt.prompt_text,
        )
        return new_articles

    def _is_url_duplicate(self, article: NewsArticle, recent_events: list[Event]) -> bool:
        """Check if article URL matches any existing event's external_id."""
        return any(e.external_id == article.url for e in recent_events)

    def _is_headline_duplicate(self, article: NewsArticle, recent_events: list[Event]) -> bool:
        """Check if normalized headline matches any existing event."""
        normalized = _normalize_headline(article.title)
        return any(_normalize_headline(e.headline) == normalized for e in recent_events)

    async def _is_semantic_duplicate(
        self, article: NewsArticle, recent_events: list[Event]
    ) -> bool:
        """Check TCR OR embedding similarity against recent events."""
        article_vec = await embed_text(self._embedding_model_client, article.title)
        items = [(e.headline, e.embedding) for e in recent_events]
        match_idx = is_embedding_duplicate(
            article.title,
            article_vec,
            items,
            DedupStrategy.TCR_OR_EMBEDDING,
            embedding_threshold=self.config.runtime.EVENT_DEDUP_SIMILARITY_THRESHOLD,
            tcr_threshold=self.config.runtime.EVENT_DEDUP_TCR_THRESHOLD,
        )
        return match_idx is not None

    # --- Event creation ---

    async def _create_events(
        self, articles: list[NewsArticle], follow_prompt: FollowPrompt
    ) -> list[Event]:
        """Create Event records and link entities for each new article."""
        events: list[Event] = []
        for article in articles:
            event = self._store_event(article, follow_prompt)
            if event is None:
                continue
            await self._store_embedding(event, article)
            await self._link_entities(event, article, follow_prompt.user)
            events.append(event)
        return events

    def _store_event(self, article: NewsArticle, follow_prompt: FollowPrompt) -> Event | None:
        """Create an Event record from a news article."""
        return self.db.events.add(
            user=follow_prompt.user,
            headline=article.title,
            summary=article.description,
            occurred_at=article.published_at,
            source_type=PennyConstants.EventSourceType.NEWS_API,
            source_url=article.url,
            external_id=article.url,
            follow_prompt_id=follow_prompt.id,
        )

    async def _store_embedding(self, event: Event, article: NewsArticle) -> None:
        """Compute and store headline embedding for future dedup."""
        if event.id is None:
            return
        vec = await embed_text(self._embedding_model_client, article.title)
        if vec is not None:
            self.db.events.update_embedding(event.id, serialize_embedding(vec))

    async def _link_entities(self, event: Event, article: NewsArticle, user: str) -> None:
        """Extract entity names from article and link to event (full mode)."""
        assert event.id is not None
        entity_names = await self._extract_entity_names(article)

        for name in entity_names:
            entity = self.db.entities.get_or_create(user, name)
            if entity and entity.id is not None:
                self.db.events.link_entity(event.id, entity.id)

    async def _extract_entity_names(self, article: NewsArticle) -> list[str]:
        """Use LLM to extract entity names from a news article."""
        content = f"Headline: {article.title}\n\n{article.description}"
        prompt = f"{Prompt.EVENT_ENTITY_EXTRACTION_PROMPT}\n\nArticle:\n{content}"
        try:
            response = await self._background_model_client.generate(
                prompt=prompt,
                format="json",
            )
            result = ExtractedEntities.model_validate_json(response.message.content)
            return result.entities
        except Exception as e:
            logger.warning("Failed to extract entities from article: %s", e)
            return []


def _normalize_headline(headline: str) -> str:
    """Normalize a headline for dedup comparison: lowercase, strip punctuation."""
    text = unicodedata.normalize("NFKD", headline.lower())
    return _HEADLINE_STRIP_RE.sub("", text).strip()
