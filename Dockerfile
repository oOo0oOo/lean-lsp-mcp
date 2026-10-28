ARG UV_VERSION=0.13.6
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:3.12.16-slim

ARG PIP_VERSION=26.2.2
# Update the URL and SHA256 together when changing PIP_VERSION.
ARG PIP_WHEEL_URL=https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl
ARG PIP_WHEEL_SHA256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70
ARG ELAN_VERSION=v4.1.2
# Architecture for elan binary. Update ELAN_SHA256 when changing.
ARG ELAN_ARCH=x86_64-unknown-linux-gnu
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

COPY --from=uv /uv /uvx /usr/local/bin/

# Bootstrap offline from a digest-verified wheel, then update the venv seed too.
COPY docker/refresh_ensurepip.py /tmp/refresh_ensurepip.py
RUN curl -fsSL -o "/tmp/pip-${PIP_VERSION}-py3-none-any.whl" "${PIP_WHEEL_URL}" \
    && echo "${PIP_WHEEL_SHA256}  /tmp/pip-${PIP_VERSION}-py3-none-any.whl" | sha256sum -c - \
    && python -m pip install --no-index --no-deps --no-cache-dir --upgrade "/tmp/pip-${PIP_VERSION}-py3-none-any.whl" \
    && python /tmp/refresh_ensurepip.py "/tmp/pip-${PIP_VERSION}-py3-none-any.whl" "${PIP_VERSION}" \
    && python -c "import ensurepip; assert ensurepip.version() == '${PIP_VERSION}'" \
    && rm -f "/tmp/pip-${PIP_VERSION}-py3-none-any.whl" /tmp/refresh_ensurepip.py

RUN mkdir -p "${ELAN_HOME}" \
    && curl -fsSL -o /tmp/elan.tar.gz "https://github.com/leanprover/elan/releases/download/${ELAN_VERSION}/elan-${ELAN_ARCH}.tar.gz" \
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
