"""Shared constants for the penny-team orchestrator."""

from __future__ import annotations

from enum import StrEnum


class Label(StrEnum):
    """GitHub issue labels — each maps to exactly one agent."""

    REQUIREMENTS = "requirements"
    SPECIFICATION = "specification"
    IN_PROGRESS = "in-progress"
    IN_REVIEW = "in-review"
    BUG = "bug"


# Labels where external state (CI checks, merge conflicts, reviews) can change
# without updating issue timestamps
LABELS_WITH_EXTERNAL_STATE = {Label.IN_REVIEW}
