"""EventAgent — polls news for followed topics, creates events."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import UTC, datetime, timedelta

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


class EventAgent(Agent):
    """Background agent that polls news for followed topics.

    Each execute() call:
    1. Checks preconditions (news tool, poll interval)
    2. Gets next FollowPrompt to poll (round-robin)
    3. Fetches articles from NewsAPI
    4. Scores by relevance (embedding similarity to topic)
    5. Deduplicates (URL → headline → embedding similarity)
    6. Ranks by relevance and caps at EVENT_MAX_PER_POLL
    7. Creates Event records for new articles
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
        scored = await self._score_relevant(articles, follow_prompt)
        deduped = await self._deduplicate(scored, follow_prompt)
        capped = self._rank_and_cap(deduped)
        events_created = await self._create_events(capped, follow_prompt)
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
            if not self._poll_interval_elapsed(prompt):
                continue
            if self._has_unannounced_events(prompt):
                continue
            return prompt
        return None

    def _has_unannounced_events(self, prompt: FollowPrompt) -> bool:
        """Check if this follow prompt has events waiting to be announced."""
        assert prompt.id is not None
        unnotified = self.db.events.get_unnotified_for_follow_prompt(prompt.id)
        if unnotified:
            logger.debug(
                "EventAgent: skipping '%s' — %d unannounced events",
                prompt.prompt_text[:50],
                len(unnotified),
            )
        return len(unnotified) > 0

    def _poll_interval_elapsed(self, prompt: FollowPrompt) -> bool:
        """Check if the fixed poll interval has elapsed since last poll."""
        if prompt.last_polled_at is None:
            return True
        last = prompt.last_polled_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        elapsed = (datetime.now(UTC) - last).total_seconds()
        return elapsed >= self.config.runtime.EVENT_POLL_INTERVAL_SECONDS

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

    async def _score_relevant(
        self, articles: list[NewsArticle], follow_prompt: FollowPrompt
    ) -> list[tuple[float, NewsArticle]]:
        """Score articles by embedding similarity to the follow prompt topic.

        Two-pass: first checks title embedding against the topic. If that
        fails, extracts topic tags from the headline via the LLM and
        checks those against the topic. This handles broad topics like
        "science" where specific article titles don't embed close to the
        bare topic word.

        Returns (score, article) pairs for articles above the threshold.
        """
        topic_vec = await embed_text(self._embedding_model_client, follow_prompt.prompt_text)
        if topic_vec is None:
            # No embedding model — pass all through with default score
            return [(1.0, article) for article in articles]

        threshold = self.config.runtime.EVENT_RELEVANCE_THRESHOLD
        scored: list[tuple[float, NewsArticle]] = []
        for article in articles:
            score = await self._relevance_score(article, topic_vec, threshold)
            if score is not None:
                scored.append((score, article))
            else:
                logger.debug("Relevance: rejected '%s' (below %.2f)", article.title[:60], threshold)

        logger.debug(
            "Relevance: %d articles → %d relevant for '%s'",
            len(articles),
            len(scored),
            follow_prompt.prompt_text,
        )
        return scored

    async def _relevance_score(
        self, article: NewsArticle, topic_vec: list[float], threshold: float
    ) -> float | None:
        """Score an article's relevance via title embedding, then tag fallback.

        Returns the cosine similarity score if above threshold, else None.
        """
        article_vec = await embed_text(self._embedding_model_client, article.title)
        if article_vec is None:
            return 1.0  # Can't embed — let it through with default score
        score = check_relevance(article_vec, topic_vec, threshold)
        if score is not None:
            return score
        # Title didn't match — try extracting topic tags from the headline
        tags_vec = await self._extract_tag_embedding(article.title)
        if tags_vec is None:
            return None
        return check_relevance(tags_vec, topic_vec, threshold)

    async def _extract_tag_embedding(self, headline: str) -> list[float] | None:
        """Extract topic tags from a headline and return their embedding."""
        prompt = Prompt.EVENT_TAG_EXTRACTION_PROMPT.format(headline=headline)
        try:
            response = await self._background_model_client.chat(
                [{"role": "user", "content": prompt}]
            )
            tags = json.loads(response.message.content.strip())
            if not isinstance(tags, list) or not tags:
                return None
            return await embed_text(self._embedding_model_client, ", ".join(tags))
        except Exception:
            logger.debug("Tag extraction failed for '%s'", headline[:60])
            return None

    # --- Dedup ---

    async def _deduplicate(
        self,
        scored_articles: list[tuple[float, NewsArticle]],
        follow_prompt: FollowPrompt,
    ) -> list[tuple[float, NewsArticle]]:
        """Three-layer dedup: URL → normalized headline → semantic (TCR OR embedding)."""
        window_days = int(self.config.runtime.EVENT_DEDUP_WINDOW_DAYS)
        recent_events = self.db.events.get_recent(follow_prompt.user, days=window_days)
        new_articles: list[tuple[float, NewsArticle]] = []

        for score, article in scored_articles:
            if self._is_url_duplicate(article, recent_events):
                continue
            if self._is_headline_duplicate(article, recent_events):
                continue
            if await self._is_semantic_duplicate(article, recent_events):
                continue
            new_articles.append((score, article))

        logger.debug(
            "Dedup: %d articles → %d new for '%s'",
            len(scored_articles),
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

    # --- Rank and cap ---

    def _rank_and_cap(self, scored_articles: list[tuple[float, NewsArticle]]) -> list[NewsArticle]:
        """Sort by relevance score descending and return top EVENT_MAX_PER_POLL."""
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        max_events = int(self.config.runtime.EVENT_MAX_PER_POLL)
        capped = scored_articles[:max_events]

        if len(scored_articles) > max_events:
            logger.debug(
                "Capped %d articles to %d (max per poll)",
                len(scored_articles),
                max_events,
            )

        return [article for _, article in capped]

    # --- Event creation ---

    async def _create_events(
        self, articles: list[NewsArticle], follow_prompt: FollowPrompt
    ) -> list[Event]:
        """Create Event records for each new article."""
        events: list[Event] = []
        for article in articles:
            event = self._store_event(article, follow_prompt)
            if event is None:
                continue
            await self._store_embedding(event, article)
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


def _normalize_headline(headline: str) -> str:
    """Normalize a headline for dedup comparison: lowercase, strip punctuation."""
    text = unicodedata.normalize("NFKD", headline.lower())
    return _HEADLINE_STRIP_RE.sub("", text).strip()
