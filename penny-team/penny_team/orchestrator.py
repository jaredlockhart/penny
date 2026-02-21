"""Penny Agent Orchestrator

Manages autonomous agent lifecycles. Each agent runs on its own schedule,
executing Claude CLI with its specific prompt.

Usage:
    python penny-team/orchestrator.py              # Run continuously
    python penny-team/orchestrator.py --once       # Run all due agents once and exit
    python penny-team/orchestrator.py --list       # Show registered agents
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["PyJWT[crypto]", "python-dotenv"]
# ///

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from github_api.api import GitHubAPI
from github_api.auth import GitHubAuth

from penny_team.base import Agent
from penny_team.constants import TeamConstants
from penny_team.monitor import MonitorAgent
from penny_team.quality import QualityAgent
from penny_team.utils.codeowners import parse_codeowners

AGENTS_DIR = Path(__file__).parent
PROJECT_ROOT = AGENTS_DIR.parent.parent
LOG_DIR = PROJECT_ROOT / TeamConstants.TEAM_LOG_DIR

logger = logging.getLogger(__name__)


def load_github_app():
    """Load GitHub App config from environment variables."""

    app_id = os.getenv(TeamConstants.ENV_APP_ID)
    key_path = os.getenv(TeamConstants.ENV_KEY_PATH)
    install_id = os.getenv(TeamConstants.ENV_INSTALL_ID)

    if app_id is None or key_path is None or install_id is None:
        return None

    key_file = Path(key_path)
    if not key_file.is_absolute():
        key_file = PROJECT_ROOT / key_file

    return GitHubAuth(
        app_id=int(app_id),
        private_key_path=key_file,
        installation_id=int(install_id),
    )


def get_agents(github_app=None) -> list[Agent]:
    """All registered agents. Add new agents here."""
    trusted_users = parse_codeowners(PROJECT_ROOT)

    if not trusted_users:
        logger.warning(
            "No trusted users found in CODEOWNERS. "
            "Agents will NOT filter issue content (prompt injection risk). "
            "Create .github/CODEOWNERS to enable filtering."
        )
        trusted: set[str] | None = None
    else:
        trusted = trusted_users

    # Trust the bot's own output — agents create issues that other agents read
    # GitHub API returns login in three formats depending on context:
    #   "slug"          — e.g. "penny-team"
    #   "slug[bot]"     — e.g. "penny-team[bot]"
    #   "app/slug"      — e.g. "app/penny-team" (issue author via gh issue view)
    if trusted is not None and github_app is not None:
        slug = github_app._fetch_slug()
        trusted.add(slug)
        trusted.add(f"{slug}{TeamConstants.BOT_SUFFIX}")
        trusted.add(f"{TeamConstants.APP_PREFIX}{slug}")

    # Create shared GitHub API client for all agents
    github_api = (
        GitHubAPI(
            github_app.get_token, TeamConstants.GITHUB_REPO_OWNER, TeamConstants.GITHUB_REPO_NAME
        )
        if github_app
        else None
    )

    agents: list[Agent] = [
        Agent(
            name=TeamConstants.AGENT_PM,
            interval_seconds=TeamConstants.PM_INTERVAL,
            timeout_seconds=TeamConstants.PM_TIMEOUT,
            required_labels=[TeamConstants.Label.REQUIREMENTS],
            github_app=github_app,
            github_api=github_api,
            trusted_users=trusted,
            post_output_as_comment=True,
            allowed_tools=[],
        ),
        Agent(
            name=TeamConstants.AGENT_ARCHITECT,
            interval_seconds=TeamConstants.ARCHITECT_INTERVAL,
            timeout_seconds=TeamConstants.ARCHITECT_TIMEOUT,
            required_labels=[TeamConstants.Label.SPECIFICATION],
            github_app=github_app,
            github_api=github_api,
            trusted_users=trusted,
            post_output_as_comment=True,
            allowed_tools=[],
        ),
        Agent(
            name=TeamConstants.AGENT_WORKER,
            interval_seconds=TeamConstants.WORKER_INTERVAL,
            timeout_seconds=TeamConstants.WORKER_TIMEOUT,
            required_labels=[
                TeamConstants.Label.IN_PROGRESS,
                TeamConstants.Label.IN_REVIEW,
                TeamConstants.Label.BUG,
            ],
            github_app=github_app,
            github_api=github_api,
            trusted_users=trusted,
            suppress_system_prompt=False,
        ),
        MonitorAgent(
            name=TeamConstants.AGENT_MONITOR,
            interval_seconds=TeamConstants.MONITOR_INTERVAL,
            timeout_seconds=TeamConstants.MONITOR_TIMEOUT,
            github_app=github_app,
            github_api=github_api,
            trusted_users=trusted,
        ),
    ]

    # Quality agent is optional — only registered when Ollama model is configured
    ollama_model = os.getenv(TeamConstants.ENV_OLLAMA_MODEL)
    if ollama_model:
        agents.append(
            QualityAgent(
                name=TeamConstants.AGENT_QUALITY,
                interval_seconds=TeamConstants.QUALITY_INTERVAL,
                timeout_seconds=TeamConstants.QUALITY_TIMEOUT,
                ollama_url=os.getenv(
                    TeamConstants.ENV_OLLAMA_URL, TeamConstants.OLLAMA_DEFAULT_URL
                ),
                ollama_model=ollama_model,
                github_app=github_app,
                github_api=github_api,
                trusted_users=trusted,
            )
        )

    return agents


LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5


def setup_logging(log_file: Path | None = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
            )
        )
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)


def _rotate_file(path: Path) -> None:
    """Rotate a file if it exceeds LOG_MAX_BYTES, keeping LOG_BACKUP_COUNT backups."""
    if not path.exists() or path.stat().st_size < LOG_MAX_BYTES:
        return
    # Shift existing backups: .log.4 -> .log.5, .log.3 -> .log.4, etc.
    for i in range(LOG_BACKUP_COUNT, 0, -1):
        src = path.with_suffix(f".log.{i}")
        dst = path.with_suffix(f".log.{i + 1}")
        if i == LOG_BACKUP_COUNT and src.exists():
            src.unlink()  # Delete oldest backup
        elif src.exists():
            src.rename(dst)
    # Rotate current file to .log.1
    path.rename(path.with_suffix(".log.1"))


def save_agent_log(
    agent_name: str,
    run_number: int,
    timestamp: datetime,
    duration: float,
    success: bool,
    output: str,
) -> None:
    """Save raw agent output to a per-agent log file."""
    log_path = LOG_DIR / f"{agent_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_file(log_path)
    with open(log_path, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Run #{run_number} at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration:.1f}s | Success: {success}\n")
        f.write(f"{'=' * 60}\n")
        f.write(output)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Penny Agent Orchestrator")
    parser.add_argument("--once", action="store_true", help="Run all due agents once and exit")
    parser.add_argument("--list", action="store_true", help="List registered agents and exit")
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Run only the named agent (e.g. 'product-manager' or 'worker')",
    )
    parser.add_argument("--log-file", type=Path, default=LOG_DIR / TeamConstants.ORCHESTRATOR_LOG)
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    # Load .env so orchestrator works without shell exports
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / TeamConstants.ENV_FILENAME)

    github_app = load_github_app()
    if github_app:
        logger.info(f"GitHub App: id={github_app.app_id}, install={github_app.installation_id}")
    else:
        logger.warning("No GitHub App configured — agents will use your personal gh auth")

    agents = get_agents(github_app)

    if args.agent:
        agents = [a for a in agents if a.name == args.agent]
        if not agents:
            logger.error(f"Unknown agent: {args.agent}")
            sys.exit(1)

    if args.list:
        for agent in agents:
            print(
                f"  {agent.name:20s}  every {agent.interval_seconds}s  prompt: {agent.prompt_path}"
            )
        return

    # Clean shutdown
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False
        # Forward SIGTERM to any running agent subprocess
        for agent in agents:
            proc = agent._process
            if proc is not None and proc.poll() is None:
                logger.info(f"Forwarding SIGTERM to [{agent.name}] subprocess (pid={proc.pid})")
                try:
                    proc.terminate()
                except ProcessLookupError:
                    logger.debug(f"Subprocess for [{agent.name}] already exited")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Orchestrator started with {len(agents)} agent(s)")
    for agent in agents:
        logger.info(f"  {agent.name} — every {agent.interval_seconds // 60}m")

    if args.once:
        for agent in agents:
            if agent.has_work():
                result = agent.run()
                save_agent_log(
                    agent.name,
                    agent.run_count,
                    result.timestamp,
                    result.duration,
                    result.success,
                    result.output,
                )
            else:
                logger.info(f"[{agent.name}] No matching issues, skipping")
        return

    # Main loop — check agents every 30s, run those that are due
    tick_seconds = 30

    while running:
        for agent in agents:
            if not running:
                break
            if agent.is_due():
                if agent.has_work():
                    result = agent.run()
                    save_agent_log(
                        agent.name,
                        agent.run_count,
                        result.timestamp,
                        result.duration,
                        result.success,
                        result.output,
                    )
                else:
                    logger.info(f"[{agent.name}] No matching issues, skipping")
                    agent.last_run = datetime.now()

        # Sleep in 1s increments so signals are responsive
        for _ in range(tick_seconds):
            if not running:
                break
            time.sleep(1)

    logger.info("Orchestrator stopped")


if __name__ == "__main__":
    main()
