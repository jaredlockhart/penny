# Use Python 3.12 slim image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY penny/ ./penny/

# Install dependencies with uv
RUN uv pip install --system websockets httpx python-dotenv pydantic sqlmodel

# Run the agent
CMD ["python", "-m", "penny.agent"]
