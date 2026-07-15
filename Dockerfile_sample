FROM python:3.14-slim

# Install uv (provides uvx for MCP servers)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install Node.js 24 from the official image instead of the distro's apt package
# (Debian ships Node 20, but @esaio/esa-mcp-server requires node >=24 and was
# unstable on Node 20 — heavy esa queries hung on EC2). Copy the binaries and
# the npm/npx module tree, then recreate the bin symlinks.
COPY --from=node:24-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:24-bookworm-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN uv pip install --system .

# Copy application
COPY main.py .
COPY slack_agent/ slack_agent/

CMD ["python", "main.py"]
