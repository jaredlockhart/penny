"""CollectorAgent — per-collection background extractor.

Replaces the bespoke preference / knowledge extractors with a single
agent class parameterized by a target collection.  Each instance reads
its target's ``extraction_prompt`` from the memory row and operates with
a tool surface scoped to writing only to that collection.

The scheduler instantiates one CollectorAgent per collection where
``Memory.extraction_prompt IS NOT NULL`` and runs each on a periodic,
idle-gated schedule.  The target's name and description are injected
into the system prompt so the model knows where to write and what the
collection is for; the user-authored ``extraction_prompt`` follows as
the body.
"""

from __future__ import annotations

from penny.agents.base import BackgroundAgent
from penny.config import Config
from penny.database import Database
from penny.database.models import Memory
from penny.llm.client import LlmClient
from penny.tools.base import Tool
from penny.tools.memory_tools import DoneTool, build_memory_tools


class CollectorAgent(BackgroundAgent):
    """Background extractor bound to a single collection."""

    def __init__(
        self,
        target: Memory,
        model_client: LlmClient,
        db: Database,
        config: Config,
        *,
        embedding_model_client: LlmClient | None = None,
        vision_model_client: LlmClient | None = None,
    ) -> None:
        if target.extraction_prompt is None:
            raise ValueError(
                f"CollectorAgent requires a target with extraction_prompt set "
                f"({target.name!r} has none)"
            )
        self._target = target
        # Set name before super().__init__ so the startup log line and
        # the registry pick up the per-instance identity.
        self.name = f"collector:{target.name}"
        super().__init__(
            model_client=model_client,
            db=db,
            config=config,
            vision_model_client=vision_model_client,
            embedding_model_client=embedding_model_client,
            system_prompt=self._compose_prompt(target),
        )

    @staticmethod
    def _compose_prompt(target: Memory) -> str:
        return (
            f"You are the collector for the `{target.name}` collection.\n"
            f"Description: {target.description}\n\n"
            f"{target.extraction_prompt}"
        )

    def get_tools(self) -> list[Tool]:
        """Scoped surface — writes only to the target collection.

        Overrides ``BackgroundAgent.get_tools()`` entirely to skip
        ``SendMessageTool``: collectors don't deliver to users, they
        only curate their own collection.  Reads stay unconstrained
        so the collector can pull context from other memories.
        """
        tools: list[Tool] = build_memory_tools(
            self.db,
            self._embedding_model_client,
            agent_name=self.name,
            scope=self._target.name,
        )
        tools.append(self._build_browse_tool(author=self.name))
        tools.append(DoneTool())
        return tools
