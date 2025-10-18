from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import MCPClient, connect_stdio_client
from tests.helpers.test_project import ensure_test_project


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def test_project_path(repo_root: Path) -> Path:
    try:
        return ensure_test_project(repo_root)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _server_environment(repo_root: Path) -> dict[str, str]:
    pythonpath_entries = [str(repo_root / "src")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)

    env: dict[str, str] = {
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        "LEAN_LOG_LEVEL": os.environ.get("LEAN_LOG_LEVEL", "ERROR"),
    }

    token = os.environ.get("LEAN_LSP_MCP_TOKEN")
    if token:
        env["LEAN_LSP_MCP_TOKEN"] = token

    return env


@pytest.fixture
def mcp_client_factory(
    repo_root: Path, test_project_path: Path
) -> Callable[[], AsyncContextManager[MCPClient]]:
    env = _server_environment(repo_root)
    _ = test_project_path  # Ensure project is prepared before starting the server

    def factory() -> AsyncContextManager[MCPClient]:
        return connect_stdio_client(
            sys.executable,
            ["-m", "lean_lsp_mcp", "--transport", "stdio"],
            env=env,
            cwd=repo_root,
        )

    return factory
