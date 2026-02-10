"""GitHub API client for agent operations.

Consolidates all GitHub API interactions (issues, PRs, workflow runs)
behind typed Pydantic models, replacing scattered gh CLI subprocess calls
with direct urllib.request calls to REST and GraphQL endpoints.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from penny_team.constants import (
    API_ISSUE_COMMENTS,
    API_JOB_LOGS,
    API_PR_REVIEW_COMMENTS,
    API_RUN_JOBS,
    API_WORKFLOW_RUNS,
    GITHUB_API,
    GITHUB_REPO_NAME,
    GITHUB_REPO_OWNER,
    GQL_ISSUES_DETAILED,
    GQL_ISSUES_LIGHTWEIGHT,
    GQL_OPEN_PRS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic models — Issue
# =============================================================================


class IssueListItem(BaseModel):
    """Lightweight issue for has_work() timestamp comparison."""

    model_config = ConfigDict(populate_by_name=True)

    number: int
    updated_at: str = Field("", alias="updatedAt")


class IssueAuthor(BaseModel):
    """Author of a GitHub issue."""

    login: str = ""


class IssueLabel(BaseModel):
    """A label on a GitHub issue."""

    name: str


class IssueComment(BaseModel):
    """A comment on a GitHub issue."""

    model_config = ConfigDict(populate_by_name=True)

    author: IssueAuthor = IssueAuthor()
    body: str = ""
    created_at: str = Field("", alias="createdAt")


class IssueDetail(BaseModel):
    """Full issue details from detailed query."""

    number: int
    title: str = ""
    body: str = ""
    author: IssueAuthor = IssueAuthor()
    labels: list[IssueLabel] = []
    comments: list[IssueComment] = []


# =============================================================================
# Pydantic models — Pull Request
# =============================================================================


class CommentAuthor(BaseModel):
    """Author of a PR comment or review."""

    login: str = ""


class PRComment(BaseModel):
    """A top-level PR comment."""

    model_config = ConfigDict(populate_by_name=True)

    author: CommentAuthor = CommentAuthor()
    body: str = ""
    created_at: str = Field("", alias="createdAt")


class PRReview(BaseModel):
    """A formal PR review."""

    model_config = ConfigDict(populate_by_name=True)

    author: CommentAuthor = CommentAuthor()
    state: str = ""
    submitted_at: str = Field("", alias="submittedAt")


class CheckStatus(BaseModel):
    """A single CI check from statusCheckRollup.

    Normalizes both CheckRun and StatusContext GraphQL types into
    a single model. For CheckRun: state=status, conclusion=conclusion.
    For StatusContext: state=state, conclusion="".
    """

    name: str
    state: str
    conclusion: str = ""


class PullRequest(BaseModel):
    """A pull request with checks, reviews, and comments."""

    model_config = ConfigDict(populate_by_name=True)

    number: int
    head_ref_name: str = Field("", alias="headRefName")
    mergeable: str = ""
    status_check_rollup: list[CheckStatus] = []
    reviews: list[PRReview] = []
    comments: list[PRComment] = []


# =============================================================================
# Pydantic models — PR Review Comments (inline code review)
# =============================================================================


class ReviewCommentUser(BaseModel):
    """Author of an inline PR review comment (REST API format)."""

    login: str = ""


class ReviewComment(BaseModel):
    """An inline review comment on a pull request (from REST API)."""

    user: ReviewCommentUser = ReviewCommentUser()
    body: str = ""
    path: str = ""
    created_at: str = ""


# =============================================================================
# Pydantic models — GitHub Actions
# =============================================================================


class WorkflowRun(BaseModel):
    """A GitHub Actions workflow run."""

    id: int


class WorkflowJob(BaseModel):
    """A job within a GitHub Actions workflow run."""

    id: int
    conclusion: str = ""


# =============================================================================
# GitHubAPI client
# =============================================================================


class GitHubAPI:
    """GitHub API client using urllib.request with token-based auth.

    Uses GraphQL for complex queries (issues with comments, PRs with
    checks/reviews) and REST for simple operations (posting comments,
    Actions API).
    """

    def __init__(self, token_provider: Callable[[], str]) -> None:
        self._get_token = token_provider

    # --- Low-level request methods ---

    def _rest_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: int = 15,
        accept: str = "application/vnd.github+json",
    ) -> Any:
        """Make a REST API request to GitHub."""
        url = f"{GITHUB_API}{path}"
        token = self._get_token()
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", accept)
        if data:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return None
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                return json.loads(raw)
            return raw.decode()

    def _graphql(
        self,
        query: str,
        variables: dict | None = None,
        timeout: int = 15,
    ) -> dict:
        """Make a GraphQL API request to GitHub."""
        body: dict[str, Any] = {"query": query}
        if variables:
            body["variables"] = variables
        result = self._rest_request("POST", "/graphql", body=body, timeout=timeout)
        if result and "errors" in result:
            errors = result["errors"]
            msg = errors[0].get("message", "Unknown GraphQL error") if errors else "Unknown"
            raise RuntimeError(f"GraphQL error: {msg}")
        return result

    # --- Issues (GraphQL) ---

    def list_issues(self, label: str, limit: int = 20) -> list[IssueListItem]:
        """List issues by label — lightweight, returns number + updatedAt only.

        Used by has_work() for fast timestamp comparison.
        """
        data = self._graphql(
            GQL_ISSUES_LIGHTWEIGHT,
            variables={
                "owner": GITHUB_REPO_OWNER,
                "repo": GITHUB_REPO_NAME,
                "label": label,
                "limit": limit,
            },
        )
        nodes = data["data"]["repository"]["issues"]["nodes"]
        return [IssueListItem.model_validate(node) for node in nodes]

    def list_issues_detailed(self, label: str, limit: int = 20) -> list[IssueDetail]:
        """List issues by label with full details including comments.

        Replaces the N+1 pattern of gh issue list + N x gh issue view
        with a single GraphQL query per label.
        """
        data = self._graphql(
            GQL_ISSUES_DETAILED,
            variables={
                "owner": GITHUB_REPO_OWNER,
                "repo": GITHUB_REPO_NAME,
                "label": label,
                "limit": limit,
            },
        )
        nodes = data["data"]["repository"]["issues"]["nodes"]
        results: list[IssueDetail] = []
        for node in nodes:
            # Flatten nested GraphQL structure into IssueDetail
            labels_raw = node.get("labels", {}).get("nodes", [])
            comments_raw = node.get("comments", {}).get("nodes", [])
            results.append(
                IssueDetail(
                    number=node["number"],
                    title=node.get("title", ""),
                    body=node.get("body", ""),
                    author=IssueAuthor(
                        login=(node.get("author") or {}).get("login", ""),
                    ),
                    labels=[IssueLabel(name=lbl["name"]) for lbl in labels_raw],
                    comments=[
                        IssueComment.model_validate(
                            {
                                "author": {"login": (c.get("author") or {}).get("login", "")},
                                "body": c.get("body", ""),
                                "createdAt": c.get("createdAt", ""),
                            }
                        )
                        for c in comments_raw
                    ],
                )
            )
        return results

    # --- Issues (REST) ---

    def comment_issue(self, number: int, body: str) -> None:
        """Post a comment on a GitHub issue.

        Raises on failure (caller handles error semantics).
        """
        path = API_ISSUE_COMMENTS.format(
            owner=GITHUB_REPO_OWNER,
            repo=GITHUB_REPO_NAME,
            number=number,
        )
        self._rest_request("POST", path, body={"body": body}, timeout=30)

    # --- PRs (GraphQL) ---

    def list_open_prs(self, limit: int = 20) -> list[PullRequest]:
        """Fetch open PRs with checks, reviews, and comments.

        The statusCheckRollup union type (CheckRun | StatusContext) is
        normalized into CheckStatus objects for uniform handling.
        """
        data = self._graphql(
            GQL_OPEN_PRS,
            variables={
                "owner": GITHUB_REPO_OWNER,
                "repo": GITHUB_REPO_NAME,
                "limit": limit,
            },
        )
        nodes = data["data"]["repository"]["pullRequests"]["nodes"]
        results: list[PullRequest] = []
        for node in nodes:
            # Extract statusCheckRollup from last commit
            checks: list[CheckStatus] = []
            commits = node.get("commits", {}).get("nodes", [])
            if commits:
                rollup = commits[0].get("commit", {}).get("statusCheckRollup")
                if rollup:
                    for ctx in rollup.get("contexts", {}).get("nodes", []):
                        typename = ctx.get("__typename", "")
                        if typename == "CheckRun":
                            checks.append(
                                CheckStatus(
                                    name=ctx.get("name", ""),
                                    state=ctx.get("status", ""),
                                    conclusion=ctx.get("conclusion") or "",
                                )
                            )
                        elif typename == "StatusContext":
                            checks.append(
                                CheckStatus(
                                    name=ctx.get("context", ""),
                                    state=ctx.get("state", ""),
                                    conclusion="",
                                )
                            )

            # Parse reviews
            reviews_raw = node.get("reviews", {}).get("nodes", [])
            reviews = [
                PRReview.model_validate(
                    {
                        "author": {"login": (r.get("author") or {}).get("login", "")},
                        "state": r.get("state", ""),
                        "submittedAt": r.get("submittedAt", ""),
                    }
                )
                for r in reviews_raw
            ]

            # Parse comments
            comments_raw = node.get("comments", {}).get("nodes", [])
            comments = [
                PRComment.model_validate(
                    {
                        "author": {"login": (c.get("author") or {}).get("login", "")},
                        "body": c.get("body", ""),
                        "createdAt": c.get("createdAt", ""),
                    }
                )
                for c in comments_raw
            ]

            results.append(
                PullRequest(
                    number=node["number"],
                    head_ref_name=node.get("headRefName", ""),  # type: ignore[unknown-argument]
                    mergeable=node.get("mergeable", ""),
                    status_check_rollup=checks,
                    reviews=reviews,
                    comments=comments,
                )
            )
        return results

    # --- PR Review Comments (REST) ---

    def list_pr_review_comments(self, pr_number: int) -> list[ReviewComment]:
        """Fetch inline review comments on a pull request."""
        path = API_PR_REVIEW_COMMENTS.format(
            owner=GITHUB_REPO_OWNER,
            repo=GITHUB_REPO_NAME,
            pr_number=pr_number,
        )
        raw_comments = self._rest_request("GET", path)
        return [ReviewComment.model_validate(c) for c in (raw_comments or [])]

    # --- Actions (REST) ---

    def list_failed_runs(self, branch: str, limit: int = 1) -> list[WorkflowRun]:
        """List recent failed workflow runs for a branch."""
        path = API_WORKFLOW_RUNS.format(
            owner=GITHUB_REPO_OWNER,
            repo=GITHUB_REPO_NAME,
        )
        params = f"?branch={urllib.parse.quote(branch)}&status=failure&per_page={limit}"
        data = self._rest_request("GET", f"{path}{params}")
        if not data or "workflow_runs" not in data:
            return []
        return [WorkflowRun(id=run["id"]) for run in data["workflow_runs"][:limit]]

    def get_failed_job_log(self, run_id: int) -> str:
        """Fetch logs from failed jobs in a workflow run.

        Equivalent to `gh run view <id> --log-failed`. Lists jobs for
        the run, finds failed ones, and fetches each job's log text.
        """
        # List jobs for this run
        jobs_path = API_RUN_JOBS.format(
            owner=GITHUB_REPO_OWNER,
            repo=GITHUB_REPO_NAME,
            run_id=run_id,
        )
        jobs_data = self._rest_request("GET", jobs_path)
        if not jobs_data or "jobs" not in jobs_data:
            return ""

        failed_jobs = [
            WorkflowJob(id=j["id"], conclusion=j.get("conclusion", ""))
            for j in jobs_data["jobs"]
            if j.get("conclusion") == "failure"
        ]

        if not failed_jobs:
            return ""

        # Fetch log for each failed job
        log_parts: list[str] = []
        for job in failed_jobs:
            try:
                log_path = API_JOB_LOGS.format(
                    owner=GITHUB_REPO_OWNER,
                    repo=GITHUB_REPO_NAME,
                    job_id=job.id,
                )
                log_text = self._rest_request("GET", log_path, timeout=30)
                if log_text:
                    log_parts.append(str(log_text))
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to fetch log for job {job.id}: {e}")

        return "\n".join(log_parts)
