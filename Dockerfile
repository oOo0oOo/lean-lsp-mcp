FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain stable \
    && ln -sf /root/.elan/bin/elan /usr/local/bin/elan \
    && ln -sf /root/.elan/bin/lean /usr/local/bin/lean \
    && ln -sf /root/.elan/bin/lake /usr/local/bin/lake

WORKDIR /app
COPY pyproject.toml README.md uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN useradd -m -u 10001 mcp
USER mcp

ENV PATH="/app/.venv/bin:${PATH}" \
    LEAN_PROJECT_PATH=/workspace \
    LEAN_MCP_STRICT_PROJECT_ROOT=true \
    LEAN_MCP_DISABLED_TOOLS=lean_run_code

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--transport", "stdio"]
