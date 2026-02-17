FROM python:3.12-slim

ARG UV_VERSION=0.8.22
ARG ELAN_VERSION=v4.1.2
ARG ELAN_SHA256=f81c2e48c1588d4612cd2c8851947898a45ac8d72748a07dff3a5694f1cf589b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    ELAN_HOME=/opt/elan

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

RUN mkdir -p "${ELAN_HOME}" \
    && curl -fsSL -o /tmp/elan.tar.gz "https://github.com/leanprover/elan/releases/download/${ELAN_VERSION}/elan-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${ELAN_SHA256}  /tmp/elan.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/elan.tar.gz -C /tmp \
    && ELAN_HOME="${ELAN_HOME}" /tmp/elan-init -y --default-toolchain none --no-modify-path \
    && rm -f /tmp/elan.tar.gz /tmp/elan-init

WORKDIR /app
COPY pyproject.toml README.md uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN useradd -m -u 10001 mcp
RUN chown -R mcp:mcp "${ELAN_HOME}" /app
USER mcp

ENV PATH="${ELAN_HOME}/bin:/app/.venv/bin:${PATH}" \
    LEAN_PROJECT_PATH=/workspace \
    LEAN_MCP_DISABLED_TOOLS=lean_run_code

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--transport", "stdio"]
