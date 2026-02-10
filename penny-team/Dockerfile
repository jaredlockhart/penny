FROM python:3.12-slim

# System dependencies: git, make, curl, gpg (for repo key imports)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        make \
        curl \
        gnupg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Node.js 22 (for Claude CLI)
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude CLI
RUN npm install -g @anthropic-ai/claude-code

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/shared:/repo/penny-team

# Snapshot the entire repo into the image (build context is project root)
COPY . /repo
RUN mkdir -p /repo/data

# Install Python deps from baked-in source
RUN cd /repo/penny && uv pip install --system . --group dev --quiet
RUN cd /repo/penny-team && uv pip install --system . --group dev --quiet

# Claude CLI refuses --dangerously-skip-permissions as root
COPY penny-team/scripts/entrypoint.sh /entrypoint.sh
RUN useradd -m -s /bin/bash agent \
    && chown -R agent:agent /repo /usr/local/lib/python3.12/ /usr/local/bin/ /usr/local/include/python3.12/ \
    && chmod +x /entrypoint.sh
USER agent

WORKDIR /repo

ENTRYPOINT ["/entrypoint.sh"]
