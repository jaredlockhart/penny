# CLAUDE.md — Penny Agent Orchestrator

## Overview

Python-based orchestrator that manages autonomous Claude CLI agents. Agents process work from GitHub Issues on a schedule, using labels as a state machine. A Monitor Agent watches production logs for errors and files bug issues automatically.

## Directory Structure

```
penny-team/
  penny_team/
    orchestrator.py     — Agent lifecycle manager, runs on schedule
    base.py             — Agent base class: wraps Claude CLI, has_work() pre-check
    constants.py        — Shared constants: Label enum, external state config
    monitor.py          — MonitorAgent: log file reader, error extraction, bug issue filing
    monitor/
      CLAUDE.md         — Monitor agent prompt (error analysis, dedup, issue creation)
    utils/
      codeowners.py     — Parses .github/CODEOWNERS for trusted usernames
      issue_filter.py   — Pre-fetches and filters issue content by trusted authors
      pr_checks.py      — Detects failing CI checks on PRs, enriches issues for worker
    product-manager/
      CLAUDE.md         — PM agent prompt (requirements gathering)
    architect/
      CLAUDE.md         — Architect agent prompt (detailed specs)
    worker/
      CLAUDE.md         — Worker agent prompt (implementation)
  tests/
    conftest.py              — Shared fixtures, helpers, and data factories
    test_codeowners.py       — CODEOWNERS parser tests (unit)
    test_orchestrator.py     — Agent registration, logging, config tests (unit)
    test_agent_shared.py     — Shared Agent base class behavior (integration)
    test_product_manager.py  — Product Manager agent flow tests (integration)
    test_architect.py        — Architect agent flow tests (integration)
    test_worker.py           — Worker agent flow + PR status + bug fix tests (integration + unit)
    test_monitor.py          — Monitor agent flow + error extraction tests (integration + unit)

  Tests strongly prefer integration style — test through agent.run() / has_work()
  entry points with MockGitHubAPI (for GitHub data) and MockPopen (for Claude CLI).
  Unit tests are only used for pure utility functions with many edge cases
  (CODEOWNERS parsing, PR matching logic).
  scripts/
    entrypoint.sh       — Claude CLI setup + orchestrator launch
  Dockerfile            — Agent container image (Python 3.12 + Node.js + Claude CLI + gh)
  pyproject.toml        — Dependencies + ruff/ty/pytest config
```

## Agent Configurations

- **Product Manager**: 300s interval, 600s timeout, label: `requirements`
- **Architect**: 300s interval, 600s timeout, label: `specification`
- **Worker**: 300s interval, 1800s timeout, labels: `in-progress`, `in-review`, `bug`
- **Monitor**: 300s interval, 600s timeout, no labels (reads log files)

## GitHub Labels Workflow

```
backlog → requirements → specification → in-progress → in-review → closed   (features)
bug → in-review → closed                                                     (bug fixes)
```

- Each label maps to exactly one agent (1:1 mapping)
- Transitions between agents are human-initiated (user moves label)
- Worker moves `in-progress` → `in-review` and `bug` → `in-review` after pushing PR (only agent-initiated transitions)
- Bug issues bypass PM and Architect entirely — Worker picks them up directly
- Bugs are prioritized over feature work (`in-progress`)

## Orchestrator Architecture

- `penny_team/orchestrator.py`: Main loop checks agents every 30s, runs those that are due; creates `GitHubAPI` instance with token provider from `GitHubApp`, passes to all agents
- `penny_team/base.py`: Agent class wraps `claude -p <prompt> --dangerously-skip-permissions --verbose --output-format stream-json`
- `--agent <name>` flag: Run a single agent instead of the full orchestrator loop
- `--once` flag: Run a single tick (all due agents) then exit
- `--list` flag: List registered agents and their configurations
- `has_work()` pre-check: Fetches issue `updatedAt` timestamps via `GitHubAPI.list_issues()`, compares to saved state in `data/penny-team/<name>.state.json` — skips Claude CLI if no issues changed since last run
- For labels with external state (e.g., `in-review`), performs full actionability check even when timestamps unchanged (CI failures, merge conflicts, review feedback can happen without issue updates)
- Per-agent processed tracking (`AgentState.processed`): allows agents sharing the same bot identity to independently track which issues they've processed
- CI fix attempt capping: Worker pauses after `MAX_CI_FIX_ATTEMPTS` (3) failed CI fix attempts without human feedback, posts comment asking for help
- State saved when all issues exhausted (`has_work()` returns False) — not after every run — to allow burning down work queues
- Fail-open design: If API calls fail, agent runs anyway
- SIGTERM forwarding for graceful shutdown of Claude CLI subprocesses

## CODEOWNERS-Based Issue Filtering

Security layer to prevent prompt injection via public GitHub issues:
- `.github/CODEOWNERS` defines trusted maintainer usernames (trust anchor)
- `penny_team/utils/codeowners.py`: Parses CODEOWNERS to extract `@username` tokens
- `penny_team/utils/issue_filter.py`: Pre-fetches issues via `GitHubAPI.list_issues_detailed()` (single GraphQL query per label), strips bodies from untrusted authors, drops comments from non-CODEOWNERS users
- Filtered issue content is injected into the agent prompt by `base.py`, so agents never need to call `gh issue view --comments` (which would bypass the filter)
- Agent CLAUDE.md prompts instruct agents to use pre-fetched content only and restrict `gh` to write operations
- Fails open without CODEOWNERS (backward compatible, logs warning)
- Requires GitHub branch protection on `main` requiring CODEOWNERS review to prevent unauthorized CODEOWNERS edits

## PR Status Detection (CI Checks & Merge Conflicts)

Worker agent automatically detects and fixes failing CI and merge conflicts on its PRs:
- `penny_team/utils/pr_checks.py`: Fetches PR check statuses and merge conflict status via `GitHubAPI.list_open_prs()` (single GraphQL query with `statusCheckRollup` union type handling), matches PRs to issues by branch naming convention (`issue-<N>-*`)
- For failing PRs, fetches error logs via `GitHubAPI.list_failed_runs()` + `get_failed_job_log()` (truncated to ~3000 chars)
- Inline review comments fetched via `GitHubAPI.list_pr_review_comments()` (REST API)
- Enriches `FilteredIssue` with `ci_status`, `ci_failure_details`, `merge_conflict`, `merge_conflict_branch`, `has_review_feedback`, and `review_comments` before prompt injection
- `pick_actionable_issue()` treats failing-CI and merge-conflict issues as actionable even when bot has last comment; prioritizes bug issues over non-bug issues
- Worker priority: merge conflicts (rebase) > failing CI (fix) > review comments > bugs > features
- Fail-open: if API calls fail, worker proceeds normally without CI/merge info

## GitHub API Module

All orchestrator GitHub interactions use the shared `github_api/` package (at repo root) — direct HTTP calls via `urllib.request` with typed Pydantic return values. The `gh` CLI is **not** used by production orchestrator code (only by Claude CLI agents inside their sandboxed sessions). The same package is also used by penny's `/bug` command.

- `GitHubAPI(token_provider, owner, repo)` — takes a callable that returns a fresh token (decoupled from `GitHubAuth`)
- `GitHubAuth(app_id, private_key_path, installation_id)` — generates GitHub App JWT installation tokens
- **GraphQL** for complex queries: issues (lightweight + detailed), PRs with checks/reviews/comments
- **REST** for simple operations: posting comments, Actions API (workflow runs/logs), PR inline review comments
- Pydantic models for all return types: `IssueListItem`, `IssueDetail`, `PullRequest`, `CheckStatus`, `ReviewComment`, `WorkflowRun`, etc.
- `statusCheckRollup` union type (`CheckRun | StatusContext`) normalized into uniform `CheckStatus` model
- Constants (GraphQL queries, REST path templates) defined in `penny_team/constants.py`

## Log Monitoring

Monitor agent automatically detects errors in penny's production logs and files bug issues:
- `penny_team/monitor.py`: `MonitorAgent` subclass that reads `data/penny.log` instead of GitHub issues
- Tracks byte offset in `data/penny-team/monitor.state.json` to only process new log content
- Python code extracts ERROR/CRITICAL lines and tracebacks from new content
- Claude CLI analyzes errors, deduplicates against existing bug issues, and creates new issues
- First run reads last 100KB of log to avoid processing entire history
- Log rotation detected by file size < saved offset (resets to 0)
- Filed issues get the `bug` label, which the Worker agent picks up via its bug fix workflow

## Docker Setup

- Agents run in Docker containers (pm, architect, worker, and monitor services in `docker-compose.yml`) with `profiles: [team]` — only started with `make up`
- Repo is snapshotted into the Docker image at build time (not volume-mounted) — agent edits don't bleed into the host working tree
- Only `data/` is volume-mounted for shared state files and logs (`data/logs/`)
- `PYTHONDONTWRITEBYTECODE=1` prevents `__pycache__` generation in containers

## Streaming Output

- Claude CLI `-p` mode buffers all output by default
- Solution: `--verbose --output-format stream-json` enables real-time streaming
- Parse JSON events: `assistant` type has text content, `tool_use` shows tool calls, `result` has final output

## Auto-Deploy

Auto-deploy runs as a Docker service (`watcher`) defined in `scripts/watcher/`:
- The watcher container polls `git fetch origin main` periodically (configurable via `DEPLOY_INTERVAL`)
- On new commits: rebuilds penny via `git archive origin/main:penny/ | docker build -t penny -` and restarts services
