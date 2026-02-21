# CLAUDE.md — Penny Project

## What Is Penny

Penny is a local-first AI agent that communicates via Signal or Discord. Users send messages, Penny searches the web via Perplexity, reasons using Ollama (local LLM), and replies in a casual, relaxed style. It runs in Docker with host networking.

Penny also has an autonomous development team (`penny-team/`) — Claude CLI agents that process GitHub Issues on a schedule, handling requirements, architecture, and implementation.

## Environment Notes

- **Logs**: Runtime logs are written to `data/penny/logs/penny.log`; agent logs are in `data/penny-team/logs/` (not docker compose logs)

## Git Workflow

Branch protection is enabled on `main`. All changes must go through pull requests.

- **Never push directly to `main`** — always create a feature branch
- Create a descriptive branch name (e.g., `add-codeowners-filtering`, `fix-scheduler-bug`)
- Commit changes to the branch, then push and create a PR
- **Use `make token` for GitHub operations** (host only): `GH_TOKEN=$(make token) gh pr create ...`
  - This generates a GitHub App installation token for authenticated `gh` CLI access
  - Agent containers already have `GH_TOKEN` set by the orchestrator — just use `gh` directly
- The user will review and merge the PR

## Documentation Maintenance

**IMPORTANT**: Always update CLAUDE.md and README.md after making significant changes to the codebase. This includes:
- New features or modules
- Architecture changes
- Configuration changes
- API changes
- Directory structure changes

Each sub-project has its own CLAUDE.md — update the relevant one(s).

## Directory Structure

```
penny/                          — Penny chat agent (Signal/Discord)
  penny/                        — Python package
  Dockerfile
  pyproject.toml
  CLAUDE.md                     — Penny-specific context
penny-team/                     — Autonomous dev team (Claude CLI agents)
  penny_team/                   — Python package
  scripts/
    entrypoint.sh               — Docker entrypoint
  Dockerfile
  pyproject.toml
  CLAUDE.md                     — Penny-team-specific context
github_api/                     — Shared GitHub API client (GraphQL + REST)
  api.py                        — GitHubAPI class (typed Pydantic return values)
  auth.py                       — GitHubAuth (App JWT token generation)
Makefile                        — Dev commands (make up, make check, make prod)
docker-compose.yml              — signal-api + penny + team services
docker-compose.override.yml     — Dev source volume overrides
scripts/
  watcher/                      — Auto-deploy service
.github/
  workflows/
    check.yml                   — CI: runs make check on push/PR to main
  CODEOWNERS                    — Trusted maintainers (used by penny-team filtering)
docs/                           — Design documents
  knowledge-system-plan.md      — Knowledge System v2 design
  knowledge-system-flows.md     — Knowledge System v2 sequence diagrams
data/                           — Runtime data (gitignored)
  penny/                        — Penny runtime data
    penny.db                    — Production database
    backups/                    — DB backups (max 5)
    logs/                       — Penny runtime logs (penny.log)
  penny-team/                   — Agent team runtime
    logs/                       — Agent logs + prompts
    state/                      — Agent state files
  private/                      — Credentials (not in repo)
```

## Running

The project runs inside Docker Compose. A top-level Makefile wraps all commands:

```bash
make up               # Start all services (penny + team) with Docker Compose
make prod             # Deploy penny only (no team, no override)
make kill             # Tear down containers and remove local images
make build            # Build the penny Docker image
make team-build       # Build the penny-team Docker image
make token            # Generate GitHub App installation token for gh CLI
make check            # Format check, lint, typecheck, and run tests (penny + penny-team)
make pytest           # Run integration tests
make fmt              # Format with ruff (penny + penny-team)
make lint             # Lint with ruff (penny + penny-team)
make fix              # Format + autofix lint issues (penny + penny-team)
make typecheck        # Type check with ty (penny + penny-team)
make migrate-test     # Test database migrations against a copy of prod DB
make migrate-validate # Check for duplicate migration number prefixes
```

On the host, dev tool commands run via `docker compose run --rm` in a temporary container (penny service for `penny/`, team service for `penny-team/`). Inside agent containers (where `LOCAL=1` is set), the same `make` targets run tools directly — no Docker-in-Docker needed.

`make prod` starts the penny service only (skips `docker-compose.override.yml` and the `team` profile). The watcher container handles auto-deploy when running the full stack via `make up`.

Prerequisites: signal-cli-rest-api on :8080 (for Signal), Ollama on :11434, Perplexity API key in .env.

## CI

GitHub Actions runs `make check` (format, lint, typecheck, tests) on every push to `main` and on pull requests. The workflow builds the Docker images and runs all checks inside containers, same as local dev. Config is in `.github/workflows/check.yml`. Both penny and penny-team code are checked in CI.

## Configuration (.env)

**Channel selection** (auto-detected if not set):
- `CHANNEL_TYPE`: "signal" or "discord"

**Signal** (required if using Signal):
- `SIGNAL_NUMBER`: Your registered Signal number
- `SIGNAL_API_URL`: signal-cli REST API endpoint (default: http://localhost:8080)

**Discord** (required if using Discord):
- `DISCORD_BOT_TOKEN`: Bot token from Discord Developer Portal
- `DISCORD_CHANNEL_ID`: Channel ID to listen to and send messages in

**Ollama**:
- `OLLAMA_API_URL`: Ollama API endpoint (default: http://host.docker.internal:11434)
- `OLLAMA_FOREGROUND_MODEL`: Fast model for user-facing messages (default: gpt-oss:20b)
- `OLLAMA_BACKGROUND_MODEL`: Smart model for background tasks (default: same as foreground). Also used by the Quality agent for response evaluation — if set, the Quality agent is registered in penny-team
- `OLLAMA_VISION_MODEL`: Vision model for image understanding (e.g., qwen3-vl). Optional; if unset, image messages get an acknowledgment response
- `OLLAMA_IMAGE_MODEL`: Image generation model (e.g., x/z-image-turbo). Optional; enables the `/draw` command when set
- `OLLAMA_MAX_RETRIES`: Retry attempts on transient Ollama errors (default: 3)
- `OLLAMA_RETRY_DELAY`: Delay in seconds between retries (default: 0.5)

**API Keys**:
- `PERPLEXITY_API_KEY`: API key for web search
- `CLAUDE_CODE_OAUTH_TOKEN`: OAuth token for Claude CLI Max plan (agent containers, via `claude setup-token`)
- `FASTMAIL_API_TOKEN`: API token for Fastmail JMAP email search (optional, enables `/email` command)

**GitHub App** (required for agent containers and `/bug` command):
- `GITHUB_APP_ID`: GitHub App ID for authenticated API access
- `GITHUB_APP_PRIVATE_KEY_PATH`: Path to GitHub App private key file
- `GITHUB_APP_INSTALLATION_ID`: GitHub App installation ID for the repository

**Behavior**:
- `MESSAGE_MAX_STEPS`: Max agent loop steps per message (default: 5)
- `EMAIL_MAX_STEPS`: Max agent loop steps for `/email` command (default: 5)
- `IDLE_SECONDS`: Global idle threshold for all background tasks (default: 300)
- `LEARN_LOOP_INTERVAL`: Interval for learn loop in seconds, runs during idle (default: 300, runtime-configurable)
- `TOOL_TIMEOUT`: Tool execution timeout in seconds (default: 60)

**Logging**:
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `LOG_FILE`: Optional path to log file
- `LOG_MAX_BYTES`: Maximum log file size before rotation (default: 10485760 / 10 MB)
- `LOG_BACKUP_COUNT`: Number of rotated backup files to keep (default: 5)
- `DB_PATH`: SQLite database location (default: /penny/data/penny/penny.db)

## Testing Philosophy

- **Strongly prefer integration tests**: Test through public entry points (e.g., `agent.run()`, `has_work()`, full message flow) rather than testing internal functions in isolation
- **Fold assertions into existing tests**: Prefer adding assertions to an existing test that covers the relevant code path over creating a new test function
- **Unit tests only for pure utility functions**: CODEOWNERS parsing, config loading, and similar pure functions with many edge cases are acceptable as unit tests
- **Mock at system boundaries**: Mock external services (Ollama, Signal, GitHub CLI, Claude CLI) but let internal code execute end-to-end
- **Never rely on real timers**: Use `wait_until(condition)` instead of `asyncio.sleep(N)` — poll for the expected side effect (DB state, message count, etc.) with a generous timeout. Fixed sleeps are fragile on slow CI and waste time on fast machines

## Design Principles

- **Python-space over model-space**: When an action can be handled deterministically in Python (e.g., posting a comment, creating a label, validating output), do it in the orchestrator rather than relying on the model to use the right tool. Model-space logic is non-deterministic and harder to test. Reserve model-space for tasks that genuinely need reasoning (writing specs, analyzing code, generating responses).
- **Pass parameters, don't swap state**: Never temporarily swap instance state (e.g., `self.db`) to change behavior. Pass the dependency as a parameter through the call chain. Refactor interfaces to accept parameters rather than mutating shared state.
- **Capture static data at build time**: Data that doesn't change during a session (e.g., git commit info) should be captured at Docker build time via build args and environment variables, not parsed at runtime via subprocess calls.
- **Initialize at startup, not in handlers**: Heavyweight setup (copying databases, creating resources) belongs at startup (entrypoint scripts, Makefile, build steps), not lazily inside message or request handlers.

## Code Style

- **Pydantic for all structured data**: All structured data (API payloads, config, internal messages) must be brokered through Pydantic models — no raw dicts
- **Constants for string literals**: All string literals must be defined as constants or enums — no magic strings in logic
- **Prefer f-strings**: Always use f-strings over string concatenation with `+`
- **Datetime columns for ordering, IDs for joining**: Always use datetime columns (`created_at`, `timestamp`, `learned_at`, etc.) for recency ordering in queries. Never use auto-increment IDs (`id`) to infer chronological order — IDs are for joins and lookups only
