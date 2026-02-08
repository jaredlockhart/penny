#!/usr/bin/env bash
set -euo pipefail

# Claude CLI requires onboarding bypass for headless use
mkdir -p ~/.claude
echo '{"hasCompletedOnboarding": true}' > ~/.claude.json

echo "[entrypoint] Starting orchestrator..."
exec python /repo/agents/orchestrator.py "$@"
