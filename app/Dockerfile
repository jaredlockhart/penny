# Use Python 3.12 slim image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and source
COPY pyproject.toml ./
COPY penny/ ./penny/

# Install package and dependencies from pyproject.toml
RUN uv pip install --system .

# Run the agent
CMD ["python", "-m", "penny.agent"]
