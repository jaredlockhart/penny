#!/usr/bin/env bash
# Watches origin/main for new commits, rebuilds penny image + restarts services.
# Runs inside a container with the Docker socket and repo mounted.

set -euo pipefail

INTERVAL="${DEPLOY_INTERVAL:-300}"
COMPOSE="docker compose -f /repo/docker-compose.yml --project-directory ${HOST_PROJECT_DIR:-/repo}"

log() { echo "[watcher $(date +%H:%M:%S)] $*"; }

cd /repo

# Symlink HOST_PROJECT_DIR to /repo so docker compose can resolve env_file
# paths inside the container while --project-directory keeps host paths for
# volume mounts resolved by the Docker daemon.
if [ -n "${HOST_PROJECT_DIR:-}" ] && [ "$HOST_PROJECT_DIR" != "/repo" ]; then
    mkdir -p "$(dirname "$HOST_PROJECT_DIR")"
    ln -sfn /repo "$HOST_PROJECT_DIR"
    log "Symlinked $HOST_PROJECT_DIR -> /repo"
fi

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

    # Update working tree (so override volume mounts serve current code)
    if ! git merge --ff-only origin/main; then
        log "Fast-forward failed, resetting to origin/main"
        git reset --hard origin/main
    fi

    # Rebuild all images and recreate changed containers
    log "Rebuilding and restarting all services..."
    GIT_MSG=$(git log -1 --pretty=%B "$CURRENT" | tr '\n' ' ' | sed 's/ *$//')
    if GIT_COMMIT="${CURRENT:0:7}" GIT_COMMIT_MESSAGE="$GIT_MSG" $COMPOSE --profile team up -d --build; then
        log "All services rebuilt and restarted"
    else
        log "Restart failed, will retry next cycle"
    fi

    LAST_REF="$CURRENT"
done
