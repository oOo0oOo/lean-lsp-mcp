from __future__ import annotations

import asyncio
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


def _child_pids(pid: int) -> list[int]:
    try:
        output = subprocess.check_output(["pgrep", "-P", str(pid)], text=True)
    except subprocess.CalledProcessError:
        return []

    result: list[int] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        child = int(line)
        result.append(child)
        result.extend(_child_pids(child))
    return result


def _process_snapshot(root_pid: int) -> list[str]:
    pids = [root_pid] + _child_pids(root_pid)
    pids = sorted(dict.fromkeys(pids))
    output = subprocess.check_output(
        ["ps", "-o", "pid=,ppid=,rss=,args=", "-p", ",".join(map(str, pids))],
        text=True,
    )
    return [line.rstrip() for line in output.splitlines() if line.strip()]


@asynccontextmanager
async def _running_http_server(
    repo_root: Path,
) -> AsyncIterator[tuple[str, str, subprocess.Popen[str]]]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    log_file = tempfile.NamedTemporaryFile(
        prefix="lean_lsp_mcp_http_", suffix=".log", delete=False
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
        yield url, log_path, process
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


async def _call_outline(url: str, file_path: Path) -> None:
    async with streamablehttp_client(url, timeout=600, sse_read_timeout=600) as (
        read_stream,
        write_stream,
        _,
    ):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            result = await session.call_tool(
                "lean_file_outline",
                {"file_path": str(file_path), "max_declarations": "5"},
            )
            text = "\n".join(
                getattr(block, "text", "") for block in result.content if hasattr(block, "text")
            )
            assert "sampleValue" in text


@pytest.mark.asyncio
async def test_streamable_http_reuses_single_client_without_future_noise(
    repo_root: Path, test_project_path: Path
) -> None:
    file_path = test_project_path / "McpTestProject.lean"

    async with _running_http_server(repo_root) as (url, log_path, process):
        for _ in range(3):
            await _call_outline(url, file_path)
            await asyncio.sleep(1)

        await asyncio.sleep(2)
        snapshot = _process_snapshot(process.pid)

        assert sum("lake serve" in line for line in snapshot) <= 1
        assert sum("lean --server" in line for line in snapshot) <= 1

    log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    assert "Future exception was never retrieved" not in log_text
