#!/usr/bin/env bash
# Watches origin/main for new commits, rebuilds penny image + restarts services.
# Runs inside a container with the Docker socket and repo mounted.

set -euo pipefail

INTERVAL="${DEPLOY_INTERVAL:-300}"
COMPOSE_FILE="/repo/docker-compose.yml"

log() { echo "[watcher $(date +%H:%M:%S)] $*"; }

cd /repo

# Record the current origin/main ref as our baseline
LAST_REF=$(git rev-parse origin/main 2>/dev/null || echo "unknown")
log "Baseline: origin/main at ${LAST_REF:0:7}"
log "Watching for changes (every ${INTERVAL}s)"

while true; do
    sleep "$INTERVAL"

    git fetch origin main --quiet 2>/dev/null || { log "fetch failed, retrying next cycle"; continue; }

    CURRENT=$(git rev-parse origin/main)

    if [ "$CURRENT" = "$LAST_REF" ]; then
        log "up to date (${CURRENT:0:7})"
        continue
    fi

    log "New commits on origin/main: ${LAST_REF:0:7} -> ${CURRENT:0:7}"

    # Rebuild penny image from origin/main without touching the working tree
    log "Rebuilding penny image..."
    if git archive origin/main:app/ | docker build -t penny - 2>&1; then
        log "Penny image rebuilt"

        # Restart penny with the new image
        log "Restarting penny..."
        docker compose -f "$COMPOSE_FILE" up -d --no-build penny

        # Restart agents (graceful — docker sends SIGTERM, waits stop_grace_period)
        log "Restarting agents..."
        if docker compose -f "$COMPOSE_FILE" --profile team restart pm worker; then
            log "All services restarted"
        else
            log "Agent restart failed, will retry on next cycle"
        fi
    else
        log "Build failed, skipping restart"
    fi

    LAST_REF="$CURRENT"
done
