#!/usr/bin/env bash
set -euo pipefail

# Install Penny's Python deps from mounted source tree (for ruff, ty, pytest)
if [ -f /repo/app/pyproject.toml ]; then
    echo "[entrypoint] Installing Penny deps from app/pyproject.toml..."
    cd /repo/app
    uv pip install --system -r pyproject.toml --group dev --quiet
    uv pip install --system --no-deps . --quiet
    cd /repo
fi

# Install orchestrator deps (PyJWT for GitHub App auth)
echo "[entrypoint] Installing orchestrator deps..."
uv pip install --system "PyJWT[crypto]" python-dotenv --quiet

echo "[entrypoint] Starting orchestrator..."
exec python /repo/agents/orchestrator.py "$@"
