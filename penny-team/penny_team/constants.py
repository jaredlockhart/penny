"""Shared constants for the penny-team orchestrator."""

from __future__ import annotations

import os
from enum import StrEnum

# =============================================================================
# Labels
# =============================================================================


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


# =============================================================================
# CLI tools
# =============================================================================

CLAUDE_CLI = os.getenv("CLAUDE_CLI", "claude")
GH_CLI = os.getenv("GH_CLI", "gh")


# =============================================================================
# Agent names
# =============================================================================

AGENT_PM = "product-manager"
AGENT_ARCHITECT = "architect"
AGENT_WORKER = "worker"


# =============================================================================
# Agent timing (seconds)
# =============================================================================

PM_INTERVAL = 300
PM_TIMEOUT = 600
ARCHITECT_INTERVAL = 300
ARCHITECT_TIMEOUT = 600
WORKER_INTERVAL = 300
WORKER_TIMEOUT = 1800


# =============================================================================
# GitHub API field names
# =============================================================================

GH_FIELD_NUMBER = "number"
GH_FIELD_UPDATED_AT = "updatedAt"


# =============================================================================
# gh CLI query field sets
# =============================================================================

GH_ISSUE_LIST_FIELDS = str(GH_FIELD_NUMBER)
GH_ISSUE_VIEW_FIELDS = "title,body,author,comments,labels"
GH_ISSUE_LIMIT = "20"
GH_PR_FIELDS = "number,headRefName,statusCheckRollup,mergeable,reviews"


# =============================================================================
# CI / PR status
# =============================================================================

CI_STATUS_PASSING = "passing"
CI_STATUS_FAILING = "failing"

# GitHub check conclusions that count as passing
PASSING_CONCLUSIONS = {"SUCCESS", "NEUTRAL", "SKIPPED", ""}

# statusCheckRollup states that mean "still running"
PENDING_STATES = {"PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED"}

# GitHub review states that indicate feedback needing attention
REVIEW_STATE_CHANGES_REQUESTED = "CHANGES_REQUESTED"

# GitHub merge status
MERGE_STATUS_CONFLICTING = "CONFLICTING"

# Max characters of failure log to include in prompt
MAX_LOG_CHARS = 3000


# =============================================================================
# Stream-JSON event types (Claude CLI --output-format stream-json)
# =============================================================================

EVENT_ASSISTANT = "assistant"
EVENT_RESULT = "result"

# Stream-JSON content block types
BLOCK_TEXT = "text"
BLOCK_TOOL_USE = "tool_use"


# =============================================================================
# File names
# =============================================================================

PROMPT_FILENAME = "CLAUDE.md"
ENV_FILENAME = ".env"
ORCHESTRATOR_LOG = "orchestrator.log"

# Standard CODEOWNERS file locations
CODEOWNERS_PATHS = [
    ".github/CODEOWNERS",
    "CODEOWNERS",
    "docs/CODEOWNERS",
]


# =============================================================================
# GitHub App
# =============================================================================

GITHUB_API = "https://api.github.com"
JWT_ALGORITHM = "RS256"
BOT_SUFFIX = "[bot]"
NOREPLY_DOMAIN = "users.noreply.github.com"

# API paths
API_APP = "/app"
API_ACCESS_TOKENS = "/app/installations/{install_id}/access_tokens"

# Environment variable keys for bot identity
ENV_GH_TOKEN = "GH_TOKEN"
ENV_GIT_AUTHOR_NAME = "GIT_AUTHOR_NAME"
ENV_GIT_AUTHOR_EMAIL = "GIT_AUTHOR_EMAIL"
ENV_GIT_COMMITTER_NAME = "GIT_COMMITTER_NAME"
ENV_GIT_COMMITTER_EMAIL = "GIT_COMMITTER_EMAIL"

# Environment variable keys for GitHub App config
ENV_APP_ID = "GITHUB_APP_ID"
ENV_KEY_PATH = "GITHUB_APP_PRIVATE_KEY_PATH"
ENV_INSTALL_ID = "GITHUB_APP_INSTALLATION_ID"
