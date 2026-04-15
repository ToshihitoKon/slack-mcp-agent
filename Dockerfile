FROM python:3.14-slim

# Install uv (provides uvx for MCP servers)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN uv pip install --system .
RUN apt update && apt install -y nodejs npm

# Copy application
COPY main.py .
COPY slack_agent/ slack_agent/

CMD ["python", "main.py"]
