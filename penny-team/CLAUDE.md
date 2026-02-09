# CLAUDE.md — Penny Agent Orchestrator

## Overview

Python-based orchestrator that manages autonomous Claude CLI agents. Agents process work from GitHub Issues on a schedule, using labels as a state machine.

## Directory Structure

```
penny-team/
  penny_team/
    orchestrator.py     — Agent lifecycle manager, runs on schedule
    base.py             — Agent base class: wraps Claude CLI, has_work() pre-check
    utils/
      github_app.py     — GitHub App JWT token generation
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
    test_worker.py           — Worker agent flow + PR status edge case tests (integration + unit)

  Tests strongly prefer integration style — test through agent.run() / has_work()
  entry points with mocked subprocess (gh CLI, Claude CLI). Unit tests are only
  used for pure utility functions with many edge cases (CODEOWNERS parsing, PR
  matching logic).

  scripts/
    entrypoint.sh       — Claude CLI setup + orchestrator launch
  Dockerfile            — Agent container image (Python 3.12 + Node.js + Claude CLI + gh)
  pyproject.toml        — Dependencies + ruff/ty/pytest config
```

## Agent Configurations

- **Product Manager**: 300s interval, 600s timeout, label: `requirements`
- **Architect**: 300s interval, 600s timeout, label: `specification`
- **Worker**: 300s interval, 1800s timeout, labels: `in-progress`, `in-review`

## GitHub Labels Workflow

```
backlog → requirements → specification → in-progress → in-review → closed
```

- Each label maps to exactly one agent (1:1 mapping)
- Transitions between agents are human-initiated (user moves label)
- Worker moves `in-progress` → `in-review` after pushing PR (only agent-initiated transition)

## Orchestrator Architecture

- `penny_team/orchestrator.py`: Main loop checks agents every 30s, runs those that are due
- `penny_team/base.py`: Agent class wraps `claude -p <prompt> --dangerously-skip-permissions --verbose --output-format stream-json`
- `--agent <name>` flag: Run a single agent instead of the full orchestrator loop
- `has_work()` pre-check: Fetches issue `updatedAt` timestamps via `gh` CLI, compares to saved state in `data/penny-team/<name>.state.json` — skips Claude CLI if no issues changed since last run
- State saved after successful runs; re-fetched to capture agent's own changes
- Fail-open design: If `gh` fails, agent runs anyway
- SIGTERM forwarding for graceful shutdown of Claude CLI subprocesses

## CODEOWNERS-Based Issue Filtering

Security layer to prevent prompt injection via public GitHub issues:
- `.github/CODEOWNERS` defines trusted maintainer usernames (trust anchor)
- `penny_team/utils/codeowners.py`: Parses CODEOWNERS to extract `@username` tokens
- `penny_team/utils/issue_filter.py`: Pre-fetches issues via `gh` JSON API, strips bodies from untrusted authors, drops comments from non-CODEOWNERS users
- Filtered issue content is injected into the agent prompt by `base.py`, so agents never need to call `gh issue view --comments` (which would bypass the filter)
- Agent CLAUDE.md prompts instruct agents to use pre-fetched content only and restrict `gh` to write operations
- Fails open without CODEOWNERS (backward compatible, logs warning)
- Requires GitHub branch protection on `main` requiring CODEOWNERS review to prevent unauthorized CODEOWNERS edits

## PR Status Detection (CI Checks & Merge Conflicts)

Worker agent automatically detects and fixes failing CI and merge conflicts on its PRs:
- `penny_team/utils/pr_checks.py`: Fetches PR check statuses and merge conflict status via `gh pr list --json statusCheckRollup,mergeable`, matches PRs to issues by branch naming convention (`issue-<N>-*`)
- For failing PRs, fetches error logs via `gh run view --log-failed` (truncated to ~3000 chars)
- Enriches `FilteredIssue` with `ci_status`, `ci_failure_details`, `merge_conflict`, and `merge_conflict_branch` before prompt injection
- `pick_actionable_issue()` treats failing-CI and merge-conflict issues as actionable even when bot has last comment
- Worker priority: merge conflicts (rebase) > failing CI (fix) > review comments
- Fail-open: if `gh` fails, worker proceeds normally without CI/merge info

## Docker Setup

- Agents run in Docker containers (pm, architect, and worker services in `docker-compose.yml`) with `profiles: [team]` — only started with `make up`
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
