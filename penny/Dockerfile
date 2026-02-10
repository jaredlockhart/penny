# Use Python 3.12 slim image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Disable __pycache__
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /penny

# Accept build args for git info (set at build time)
ARG GIT_COMMIT=unknown
ARG GIT_COMMIT_MESSAGE=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_COMMIT_MESSAGE=${GIT_COMMIT_MESSAGE}

# Copy dependency files and install dependencies (cached layer)
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml --group dev

# Copy source
COPY penny/ ./penny/

# Install the package itself
RUN uv pip install --system --no-deps .

# Copy and setup entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run the agent
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "penny.penny"]
