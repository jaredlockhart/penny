#!/usr/bin/env bash
set -euo pipefail

# Claude CLI requires onboarding bypass for headless use
mkdir -p ~/.claude
echo '{"hasCompletedOnboarding": true}' > ~/.claude.json

# Configure git to authenticate via GH_TOKEN (set by orchestrator)
git config --global credential.helper '!gh auth git-credential'
git config --global user.name "penny-team[bot]"
git config --global user.email "penny-team[bot]@users.noreply.github.com"

echo "[entrypoint] Starting orchestrator..."
exec python /repo/penny-team/orchestrator.py "$@"
