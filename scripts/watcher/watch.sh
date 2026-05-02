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

    # Build penny image from origin/main via git archive (never touches working tree)
    # origin/main:penny archives penny/ contents at root, matching what the Dockerfile expects
    log "Rebuilding and restarting penny..."
    GIT_MSG=$(git log -1 --pretty=%B "$CURRENT" | tr '\n' ' ' | sed 's/ *$//')
    if git archive origin/main:penny \
        | docker build -t penny:latest \
            --build-arg "GIT_COMMIT=${CURRENT:0:7}" \
            --build-arg "GIT_COMMIT_MESSAGE=$GIT_MSG" \
            -f Dockerfile - \
        && $COMPOSE up -d --no-build --no-deps penny; then
        log "penny rebuilt and restarted"
        LAST_REF="$CURRENT"
    else
        log "Restart failed, will retry next cycle"
    fi
done
