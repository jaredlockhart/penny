#!/usr/bin/env bash
# Tails logs and auto-restarts Penny when main has new commits.
# Called by `make deploy` after initial startup.
#
# Usage:
#   ./scripts/deploy-watch.sh              # Check for updates every 5 minutes
#   ./scripts/deploy-watch.sh 60           # Check every 60 seconds

set -euo pipefail

INTERVAL="$1"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

log() { echo "[deploy $(date +%H:%M:%S)] $*"; }

# Cleanup on exit
cleanup() {
    log "shutting down..."
    kill "$WATCH_PID" 2>/dev/null || true
    docker compose down
    log "stopped"
    exit
}
trap cleanup INT TERM

log "watching for updates (every ${INTERVAL}s)"

# Background: poll git for changes and restart on new commits
(
    while true; do
        sleep "$INTERVAL"
        git fetch origin main --quiet 2>/dev/null || continue

        LOCAL=$(git rev-parse HEAD)
        REMOTE=$(git rev-parse origin/main)

        if [ "$LOCAL" = "$REMOTE" ]; then
            log "up to date (${LOCAL:0:7})"
            continue
        fi

        log "new commits detected ($(git log --oneline HEAD..origin/main | head -3))"
        git pull origin main --ff-only || { log "pull failed, skipping"; continue; }

        log "rebuilding and restarting..."
        docker compose down
        docker compose up -d --build
        log "restarted"
    done
) &
WATCH_PID=$!

# Foreground: tail logs (restarts if containers are recreated)
while true; do
    docker compose logs -f --tail 50 2>/dev/null || true
    sleep 2
done
