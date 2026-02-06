"""Penny Agent Orchestrator

Manages autonomous agent lifecycles. Each agent runs on its own schedule,
executing Claude CLI with its specific prompt.

Usage:
    python agents/orchestrator.py              # Run continuously
    python agents/orchestrator.py --once       # Run all due agents once and exit
    python agents/orchestrator.py --list       # Show registered agents
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from base import Agent
from codeowners import parse_codeowners

AGENTS_DIR = Path(__file__).parent
PROJECT_ROOT = AGENTS_DIR.parent
LOG_DIR = AGENTS_DIR / "logs"

logger = logging.getLogger(__name__)


def get_agents() -> list[Agent]:
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

    return [
        Agent(
            name="product-manager",
            prompt_path=AGENTS_DIR / "product-manager" / "CLAUDE.md",
            interval_seconds=300,
            timeout_seconds=600,
            required_labels=["idea", "draft"],
            trusted_users=trusted,
        ),
        Agent(
            name="worker",
            prompt_path=AGENTS_DIR / "worker" / "CLAUDE.md",
            interval_seconds=300,
            timeout_seconds=1800,
            required_labels=["approved", "in-progress"],
            trusted_users=trusted,
        ),
        # Future agents:
        # Agent(name="quality", ...),
    ]


def setup_logging(log_file: Path | None = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)


def save_agent_log(agent_name: str, run_number: int, timestamp: datetime, duration: float, success: bool, output: str) -> None:
    """Save raw agent output to a per-agent log file."""
    log_path = LOG_DIR / f"{agent_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--log-file", type=Path, default=LOG_DIR / "orchestrator.log")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    agents = get_agents()

    if args.list:
        for agent in agents:
            print(f"  {agent.name:20s}  every {agent.interval_seconds}s  prompt: {agent.prompt_path}")
        return

    # Clean shutdown
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Orchestrator started with {len(agents)} agent(s)")
    for agent in agents:
        logger.info(f"  {agent.name} — every {agent.interval_seconds // 60}m")

    if args.once:
        for agent in agents:
            if agent.has_work():
                result = agent.run()
                save_agent_log(agent.name, agent.run_count, result.timestamp, result.duration, result.success, result.output)
            else:
                logger.info(f"[{agent.name}] No matching issues, skipping")
        return

    # Main loop — check agents every 30s, run those that are due
    TICK_SECONDS = 30

    while running:
        for agent in agents:
            if not running:
                break
            if agent.is_due():
                if agent.has_work():
                    result = agent.run()
                    save_agent_log(agent.name, agent.run_count, result.timestamp, result.duration, result.success, result.output)
                else:
                    logger.info(f"[{agent.name}] No matching issues, skipping")
                    agent.last_run = datetime.now()

        # Sleep in 1s increments so signals are responsive
        for _ in range(TICK_SECONDS):
            if not running:
                break
            time.sleep(1)

    logger.info("Orchestrator stopped")


if __name__ == "__main__":
    main()
