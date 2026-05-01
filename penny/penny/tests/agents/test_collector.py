"""Unit tests for CollectorAgent — per-collection background extractor.

Construction-level tests only.  Full lifecycle integration (scheduling,
log → write → cursor advance) lands when the scheduler is wired in
phase 5; these assertions cover the agent's identity, prompt assembly,
target-binding validation, and tool-surface scoping in isolation.
"""

from __future__ import annotations

import pytest

from penny.agents.collector import CollectorAgent
from penny.database import Database
from penny.database.models import Memory
from penny.llm.client import LlmClient


def _target() -> Memory:
    return Memory(
        name="prague-trip",
        type="collection",
        description="Sights, restaurants, and bars for Jared's Prague trip",
        recall="relevant",
        archived=False,
        extraction_prompt="Read recent chat and browse logs; extract Prague spots.",
    )


def _llm_client() -> LlmClient:
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def test_collector_name_is_collection_scoped(test_config, tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.create_tables()

    agent = CollectorAgent(
        target=_target(),
        model_client=_llm_client(),
        db=db,
        config=test_config,
    )

    assert agent.name == "collector:prague-trip"


def test_collector_system_prompt_includes_target_and_extraction_prompt(test_config, tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.create_tables()

    agent = CollectorAgent(
        target=_target(),
        model_client=_llm_client(),
        db=db,
        config=test_config,
    )

    assert "`prague-trip`" in agent.system_prompt
    assert "Sights, restaurants, and bars" in agent.system_prompt
    assert "Read recent chat and browse logs; extract Prague spots." in agent.system_prompt


def test_collector_rejects_target_without_extraction_prompt(test_config, tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.create_tables()
    bare = Memory(
        name="notified-thoughts",
        type="collection",
        description="Thoughts already shared",
        recall="relevant",
        archived=False,
        extraction_prompt=None,
    )

    with pytest.raises(ValueError, match="extraction_prompt"):
        CollectorAgent(
            target=bare,
            model_client=_llm_client(),
            db=db,
            config=test_config,
        )


def test_collector_tool_surface_is_scoped_to_target(test_config, tmp_path):
    """Collector's tool surface excludes metadata + cross-collection writes,
    and the writes it does have are scope-pinned to the target collection.
    """
    db = Database(str(tmp_path / "t.db"))
    db.create_tables()
    agent = CollectorAgent(
        target=_target(),
        model_client=_llm_client(),
        db=db,
        config=test_config,
    )

    names = {tool.name for tool in agent.get_tools()}

    # No metadata or cross-collection writes
    forbidden = {
        "collection_create",
        "log_create",
        "collection_archive",
        "collection_unarchive",
        "collection_move",
        "log_append",
        "send_message",
    }
    assert names.isdisjoint(forbidden)

    # The three scoped writes plus done plus reads plus browse
    assert {"collection_write", "collection_update", "collection_delete_entry"} <= names
    assert "done" in names
    assert "browse" in names
