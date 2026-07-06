"""Centralized configuration: all lean-lsp-mcp environment variables in one place.

This module is the single source of truth for the environment variables the
server reads. Each value is resolved live from ``os.environ`` on every call so
that runtime changes (and test ``monkeypatch.setenv``) take effect — nothing is
cached at import time.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# --- Environment variable names (single source of truth) ---
ACTIVE_TRANSPORT_ENV = "LEAN_LSP_MCP_ACTIVE_TRANSPORT"
MAX_OPEN_FILES_ENV = "LEAN_LSP_MAX_OPEN_FILES"
TEST_MODE_ENV = "LEAN_LSP_TEST_MODE"
PROJECT_PATH_ENV = "LEAN_PROJECT_PATH"
AUTH_TOKEN_ENV = "LEAN_LSP_MCP_TOKEN"
BUILD_CONCURRENCY_ENV = "LEAN_BUILD_CONCURRENCY"
LOG_FILE_CONFIG_ENV = "LEAN_LOG_FILE_CONFIG"
LOG_LEVEL_ENV = "LEAN_LOG_LEVEL"
DISABLED_TOOLS_ENV = "LEAN_MCP_DISABLED_TOOLS"
TOOL_DESCRIPTIONS_ENV = "LEAN_MCP_TOOL_DESCRIPTIONS"
INSTRUCTIONS_ENV = "LEAN_MCP_INSTRUCTIONS"
LOOGLE_LOCAL_ENV = "LEAN_LOOGLE_LOCAL"
LOOGLE_CACHE_DIR_ENV = "LEAN_LOOGLE_CACHE_DIR"
LOOGLE_URL_ENV = "LOOGLE_URL"
LOOGLE_HEADERS_ENV = "LOOGLE_HEADERS"
REPL_ENV = "LEAN_REPL"
REPL_PATH_ENV = "LEAN_REPL_PATH"
REPL_TIMEOUT_ENV = "LEAN_REPL_TIMEOUT"
REPL_MEM_MB_ENV = "LEAN_REPL_MEM_MB"
STATE_SEARCH_URL_ENV = "LEAN_STATE_SEARCH_URL"
HAMMER_URL_ENV = "LEAN_HAMMER_URL"

# --- Default backends (the shared public services) ---
DEFAULT_LOOGLE_URL = "https://loogle.lean-lang.org"
DEFAULT_STATE_SEARCH_URL = "https://premise-search.com"
DEFAULT_HAMMER_URL = "http://leanpremise.net"

_TRUTHY = ("1", "true", "yes")


# --- Transport / project ---
def active_transport() -> str:
    return os.environ.get(ACTIVE_TRANSPORT_ENV, "stdio").strip().lower() or "stdio"


def project_path() -> str:
    return os.environ.get(PROJECT_PATH_ENV, "").strip()


def auth_token() -> str | None:
    return os.environ.get(AUTH_TOKEN_ENV)


def test_mode() -> bool:
    return bool(os.environ.get(TEST_MODE_ENV))


def max_open_files() -> int:
    raw_value = os.environ.get(MAX_OPEN_FILES_ENV, "4")
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%s, defaulting to 4.", MAX_OPEN_FILES_ENV, raw_value)
        return 4
    if value < 1:
        logger.warning("Invalid %s=%s, defaulting to 4.", MAX_OPEN_FILES_ENV, raw_value)
        return 4
    return value


PREWARM_FILES_ENV = "LEAN_MCP_PREWARM_FILES"


def prewarm_files() -> list[str]:
    """Project-relative Lean files to open/elaborate in the background at
    server startup (comma-separated). Warmup overlaps the agent's first
    reading/planning phase instead of landing on its first tool call."""
    raw = os.environ.get(PREWARM_FILES_ENV, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


SCRATCH_POOL_ENV = "LEAN_MCP_SCRATCH_SLOTS"


def scratch_pool_size() -> int:
    raw_value = os.environ.get(SCRATCH_POOL_ENV, "2")
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%s, defaulting to 2.", SCRATCH_POOL_ENV, raw_value)
        return 2
    return value if value >= 1 else 2


# --- Build / logging ---
def build_concurrency() -> str:
    mode = os.environ.get(BUILD_CONCURRENCY_ENV, "allow").strip().lower()
    if mode not in {"allow", "cancel", "share"}:
        logger.warning(
            "Invalid %s=%s, defaulting to allow.", BUILD_CONCURRENCY_ENV, mode
        )
        mode = "allow"
    return mode


def log_file_config() -> str | None:
    return os.environ.get(LOG_FILE_CONFIG_ENV, None)


def log_level() -> str:
    return os.environ.get(LOG_LEVEL_ENV, "INFO")


# --- Tool configuration overrides ---
def disabled_tools_raw() -> str | None:
    return os.environ.get(DISABLED_TOOLS_ENV)


def tool_descriptions_raw() -> str:
    return os.environ.get(TOOL_DESCRIPTIONS_ENV, "").strip()


def instructions_override() -> str | None:
    return os.environ.get(INSTRUCTIONS_ENV)


# --- Loogle ---
def loogle_local_enabled() -> bool:
    return os.environ.get(LOOGLE_LOCAL_ENV, "").lower() in _TRUTHY


def loogle_cache_dir() -> str | None:
    return os.environ.get(LOOGLE_CACHE_DIR_ENV)


def loogle_url() -> str:
    return os.environ.get(LOOGLE_URL_ENV, DEFAULT_LOOGLE_URL)


def loogle_headers_raw() -> str | None:
    return os.environ.get(LOOGLE_HEADERS_ENV)


# --- Remote search backends ---
# Client-side limits for shared public services: tool -> (max requests, seconds).
# Single source of truth: the @rate_limited decorators and the generated
# INSTRUCTIONS text both read from here.
RATE_LIMITS: dict[str, tuple[int, int]] = {
    "leansearch": (90, 30),
    "loogle": (3, 30),
    "leanfinder": (10, 30),
    "lean_state_search": (6, 30),
    "hammer_premise": (6, 30),
}

STATE_SEARCH_REV_ENV = "LEAN_STATE_SEARCH_REV"
DEFAULT_STATE_SEARCH_REV = "v4.22.0"

LEANFINDER_URL_ENV = "LEAN_FINDER_URL"
DEFAULT_LEANFINDER_URL = (
    "https://bxrituxuhpc70w8w.us-east-1.aws.endpoints.huggingface.cloud"
)


def state_search_url() -> str:
    return os.environ.get(STATE_SEARCH_URL_ENV, DEFAULT_STATE_SEARCH_URL)


def state_search_rev() -> str:
    return os.environ.get(STATE_SEARCH_REV_ENV, DEFAULT_STATE_SEARCH_REV)


def leanfinder_url() -> str:
    return os.environ.get(LEANFINDER_URL_ENV, DEFAULT_LEANFINDER_URL)


def hammer_url() -> str:
    return os.environ.get(HAMMER_URL_ENV, DEFAULT_HAMMER_URL)


def is_custom_backend(env_var: str, default_url: str) -> bool:
    """True when the user configured a self-hosted backend for a tool.

    A custom (non-default) URL means requests do not hit the shared public
    service, so the rate limit no longer applies.
    """
    configured = os.environ.get(env_var, "").strip()
    return bool(configured) and configured.rstrip("/") != default_url.rstrip("/")


# --- REPL ---
def repl_enabled() -> bool:
    return os.environ.get(REPL_ENV, "").lower() in _TRUTHY


def repl_path() -> str | None:
    return os.environ.get(REPL_PATH_ENV)


def _int_env(env_var: str, default: int) -> int:
    raw_value = os.environ.get(env_var)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s=%s, defaulting to %d.", env_var, raw_value, default
        )
        return default


def repl_timeout() -> int:
    return _int_env(REPL_TIMEOUT_ENV, 60)


def repl_mem_mb() -> int:
    return _int_env(REPL_MEM_MB_ENV, 8192)
