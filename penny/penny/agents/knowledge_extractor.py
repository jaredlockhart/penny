"""KnowledgeExtractorAgent — turns browse results into durable knowledge.

Reads new entries from ``browse-results`` via ``log_read_next``,
summarizes each page, and writes summaries to the ``knowledge``
collection.  User-independent — runs once per cycle regardless of
which user is active.  Cursor commits only on a clean ``done()``
exit.
"""

from __future__ import annotations

from penny.agents.base import BackgroundAgent
from penny.prompts import Prompt


class KnowledgeExtractorAgent(BackgroundAgent):
    """Background worker that builds the knowledge base from browse output."""

    name = "knowledge-extractor"
    system_prompt = Prompt.KNOWLEDGE_EXTRACTOR_SYSTEM_PROMPT
