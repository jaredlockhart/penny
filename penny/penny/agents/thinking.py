"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler after extraction. Each cycle is a continuous
thinking loop where Penny thinks out loud, uses tools, and accumulates
reasoning. At the end, the monologue is summarized and stored as a thought.
"""

from __future__ import annotations

import logging
import random

from penny.agents.base import Agent
from penny.agents.models import ChatMessage, MessageRole
from penny.constants import PennyConstants
from penny.ollama.embeddings import cosine_similarity, deserialize_embedding
from penny.prompts import Prompt

logger = logging.getLogger(__name__)

# How many top-scored topics to randomly pick from
PREFERRED_TOPIC_POOL_SIZE = 5

# Probability of a free-thinking cycle (no seed, no context, just vibes)
FREE_THINKING_PROBABILITY = 1 / 3


class ThinkingAgent(Agent):
    """Autonomous inner monologue — Penny's conscious mind.

    Each cycle picks ONE random seed topic from history
    to focus on, keeping thinking rotating across interests.

    System message (slim — profile, entities, thoughts only)::

        [system]
        <current datetime>
        <Prompt.PENNY_IDENTITY>
        <user profile>

        ## Relevant Knowledge
        - <top K entities by embedding similarity to accumulated monologue>

        ## Recent Background Thinking
        <today's thought summaries — used to avoid repetition>

        <Prompt.THINKING_SYSTEM_PROMPT with {tools} listing>

    Thinking loop::

        [user]       Think about {seed topic} and explore interesting related topics.
        [assistant]  <inner monologue text>          ← captured
        [user]       keep exploring
        [assistant]  <tool call: search(...)>         ← tool executed
        [tool]       <search results>
        [assistant]  <inner monologue reflecting>     ← captured
        ...
        <INNER_MONOLOGUE_MAX_STEPS iterations. Entity context rebuilt
        each text step, anchored to accumulated monologue.>

    Summary step::

        [system]  <Prompt.SUMMARIZE_TO_PARAGRAPH>
        [user]    <all inner monologue text joined with --->
        <background model, no tools. Result stored in db.thoughts.>

    Seed topic sources:
        - history entries (daily conversation topic bullets)
    """

    THOUGHT_CONTEXT_LIMIT = 50

    name = "inner_monologue"

    def __init__(self, **kwargs: object) -> None:
        kwargs["system_prompt"] = Prompt.THINKING_SYSTEM_PROMPT
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.max_steps = int(self.config.runtime.INNER_MONOLOGUE_MAX_STEPS)
        self._inner_monologue: list[str] = []
        self._free_thinking: bool = False

    # ── Execution hooks ──────────────────────────────────────────────────

    async def get_prompt(self, user: str) -> str | None:
        """Pick a seed topic or let Penny free-think (~1/3 of the time)."""
        self._inner_monologue = []
        self._free_thinking = False

        if random.random() < FREE_THINKING_PROBABILITY:
            logger.info("Free thinking cycle for %s", user)
            self._free_thinking = True
            return Prompt.THINKING_FREE

        topics = self._collect_topics(user)
        if not topics:
            logger.info("No seed topics for %s, browsing news", user)
            return Prompt.THINKING_BROWSE_NEWS

        seed = await self._pick_preferred_topic(user, topics)
        logger.info("Thinking seed: %s", seed)
        return Prompt.THINKING_SEED.format(seed=seed)

    def _collect_topics(self, user: str) -> list[str]:
        """Gather topic lines from recent history entries."""
        topics: list[str] = []
        limit = int(self.config.runtime.HISTORY_CONTEXT_LIMIT)
        for entry in self.db.history.get_recent(user, "daily", limit=limit):
            for line in entry.topics.splitlines():
                topic = line.strip().lstrip("- ").strip()
                if topic:
                    topics.append(topic)
        return topics

    async def _pick_preferred_topic(self, user: str, topics: list[str]) -> str:
        """Score topics by preference affinity and pick from the top pool."""
        likes, dislikes = self._load_preference_vectors(user)
        if not likes and not dislikes:
            return random.choice(topics)
        if not self._embedding_model_client:
            return random.choice(topics)

        scored = await self._score_topics(topics, likes, dislikes)
        if not scored:
            return random.choice(topics)

        scored.sort(key=lambda pair: pair[1], reverse=True)
        pool = scored[:PREFERRED_TOPIC_POOL_SIZE]
        return random.choice(pool)[0]

    def _load_preference_vectors(self, user: str) -> tuple[list[list[float]], list[list[float]]]:
        """Load like and dislike embedding vectors from the preference store."""
        prefs = self.db.preferences.get_with_embeddings(user)
        likes: list[list[float]] = []
        dislikes: list[list[float]] = []
        for p in prefs:
            if not p.embedding:
                continue
            vec = deserialize_embedding(p.embedding)
            if p.valence == PennyConstants.PreferenceValence.POSITIVE:
                likes.append(vec)
            elif p.valence == PennyConstants.PreferenceValence.NEGATIVE:
                dislikes.append(vec)
        return likes, dislikes

    async def _score_topics(
        self,
        topics: list[str],
        likes: list[list[float]],
        dislikes: list[list[float]],
    ) -> list[tuple[str, float]]:
        """Embed each topic and compute preference sentiment score."""
        scored: list[tuple[str, float]] = []
        for topic in topics:
            vec = await self._embed_topic(topic)
            if vec is None:
                continue
            score = self._compute_sentiment_score(vec, likes, dislikes)
            scored.append((topic, score))
        return scored

    @staticmethod
    def _compute_sentiment_score(
        topic_vec: list[float],
        likes: list[list[float]],
        dislikes: list[list[float]],
    ) -> float:
        """Score = avg similarity to likes - avg similarity to dislikes."""
        like_score = 0.0
        if likes:
            like_score = sum(cosine_similarity(topic_vec, lv) for lv in likes) / len(likes)
        dislike_score = 0.0
        if dislikes:
            dislike_score = sum(cosine_similarity(topic_vec, dv) for dv in dislikes) / len(dislikes)
        return like_score - dislike_score

    async def _embed_topic(self, text: str) -> list[float] | None:
        """Embed a single topic string."""
        if not self._embedding_model_client:
            return None
        try:
            vecs = await self._embedding_model_client.embed(text)
            return vecs[0]
        except Exception as e:
            logger.warning("Failed to embed topic: %s", e)
            return None

    async def get_context(self, user: str) -> str:
        """Slim context — profile, entities, and thoughts only.

        Free-thinking cycles get no context so Penny explores freely.
        """
        if self._free_thinking:
            return ""
        sections: list[str | None] = [
            self._build_profile_context(user, None),
            await self._build_entity_context(user, None),
            self._build_thought_context(user),
        ]
        return "\n\n".join(s for s in sections if s)

    async def after_run(self, user: str) -> bool:
        """Summarize the inner monologue and store as a thought."""
        if not self._inner_monologue:
            return False
        combined = "\n\n---\n\n".join(self._inner_monologue)
        summary = await self._summarize_text(combined, Prompt.SUMMARIZE_TO_PARAGRAPH)
        if summary:
            self.db.thoughts.add(user, summary)
            logger.info("[inner_monologue] %s", summary[:200])
        return True

    # ── Loop hooks ─────────────────────────────────────────────────────────

    def on_response(self, response) -> None:
        """Capture text content from every response (even tool-call ones)."""
        content = response.content.strip()
        if content:
            self._inner_monologue.append(content)

    async def handle_text_step(
        self, response, messages: list[dict], step: int, is_final: bool
    ) -> bool:
        """Inject 'keep exploring' continuation to drive the thinking loop."""
        if is_final:
            return False
        content = response.content.strip()
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content).to_dict())
        assert self._current_user is not None
        anchor = "\n".join(self._inner_monologue)
        sections: list[str | None] = [
            self._build_profile_context(self._current_user, anchor),
            await self._build_entity_context(self._current_user, anchor),
            self._build_thought_context(self._current_user),
        ]
        context_text = "\n\n".join(s for s in sections if s)
        fresh = self._build_messages(
            prompt="",
            history=None,
            system_prompt=Prompt.THINKING_SYSTEM_PROMPT,
            context=context_text,
        )
        messages[0] = fresh[0]
        messages.append(ChatMessage(role=MessageRole.USER, content="keep exploring").to_dict())
        return True
