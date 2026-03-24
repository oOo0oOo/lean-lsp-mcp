from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tests.helpers.mcp_client import result_text


_ALLOWED_TOOLS_HEADER = "X-Lean-LSP-Allowed-Tools"


def _server_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(repo_root / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["LEAN_LOG_LEVEL"] = "WARNING"
    env["LEAN_LSP_TEST_MODE"] = "1"
    return env


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("HTTP server did not start in time")


@asynccontextmanager
async def _running_http_server(
    repo_root: Path,
) -> AsyncIterator[str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    log_file = tempfile.NamedTemporaryFile(
        prefix="lean_lsp_mcp_http_policy_", suffix=".log", delete=False
    )
    log_path = log_file.name
    log_file.close()

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lean_lsp_mcp",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=repo_root,
        env=_server_env(repo_root),
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )

    try:
        _wait_for_server(port)
        yield url
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(1)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def _list_tools(url: str, headers: dict[str, str] | None = None) -> list[str]:
    async with streamablehttp_client(url, headers=headers, timeout=600, sse_read_timeout=600) as (
        read_stream,
        write_stream,
        _,
    ):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            result = await session.list_tools()
            return [tool.name for tool in result.tools]


async def _call_outline(
    url: str,
    file_path: Path,
    headers: dict[str, str] | None = None,
):
    async with streamablehttp_client(url, headers=headers, timeout=600, sse_read_timeout=600) as (
        read_stream,
        write_stream,
        _,
    ):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            return await session.call_tool(
                "lean_file_outline",
                {"file_path": str(file_path), "max_declarations": "5"},
            )


@pytest.mark.asyncio
async def test_http_session_filters_listed_tools(repo_root: Path, test_project_path: Path) -> None:
    file_path = test_project_path / "McpTestProject.lean"
    headers = {
        _ALLOWED_TOOLS_HEADER: "lean_file_outline,lean_diagnostic_messages",
    }

    async with _running_http_server(repo_root) as url:
        restricted_tools = await _list_tools(url, headers=headers)
        unrestricted_tools = await _list_tools(url)

        assert "lean_file_outline" in restricted_tools
        assert "lean_diagnostic_messages" in restricted_tools
        assert "lean_run_code" not in restricted_tools
        assert "lean_build" not in restricted_tools
        assert "lean_run_code" in unrestricted_tools

        result = await _call_outline(url, file_path, headers=headers)
        assert not result.isError
        assert "sampleValue" in result_text(result)


@pytest.mark.asyncio
async def test_http_session_rejects_disallowed_tool_calls(repo_root: Path, test_project_path: Path) -> None:
    file_path = test_project_path / "McpTestProject.lean"
    headers = {
        _ALLOWED_TOOLS_HEADER: "lean_diagnostic_messages",
    }

    async with _running_http_server(repo_root) as url:
        result = await _call_outline(url, file_path, headers=headers)
        assert result.isError
        assert "not available in this session" in result_text(result)
