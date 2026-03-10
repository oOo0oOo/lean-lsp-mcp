"""Test max_wait timeout across transports and file types.

8-way matrix: (stdio | HTTP) × (with fix | without fix) × (stuck | normal file)

"With fix" = max_wait set. "Without fix" = max_wait not set (no timeout).
For "without fix" on the stuck file, we use a short bash timeout to prove it blocks.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

# ── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
QI_PROJECT = Path("/home/maor/Desktop/git/QI-holevo/QuantumInformation")

# These files must exist (created during the stuck-worker investigation)
STUCK_FILE = str(QI_PROJECT / "QuantumInformation" / "ReproSlow.lean")
NORMAL_FILE = str(QI_PROJECT / "QuantumInformation" / "ReproStuck.lean")

# Timeouts
MAX_WAIT_FIX = 30.0  # short max_wait for "with fix" tests
NO_FIX_BLOCK_TIMEOUT = 60  # seconds to wait before concluding "no fix" blocks


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["LEAN_LOG_LEVEL"] = "WARNING"
    return env


# ── MCP client helpers ──────────────────────────────────────────────────────


@asynccontextmanager
async def connect_stdio() -> AsyncIterator[ClientSession]:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "lean_lsp_mcp", "--transport", "stdio"],
        env=_server_env(),
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(server) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            yield session


@asynccontextmanager
async def connect_http() -> AsyncIterator[ClientSession]:
    """Start MCP server with streamable-http, connect to it."""
    # Find a free port
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "lean_lsp_mcp",
            "--transport", "streamable-http",
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
        env=_server_env(),
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = f"http://127.0.0.1:{port}/mcp"

    # Wait for server to start
    for _ in range(30):
        await asyncio.sleep(0.5)
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                # Just check if port is listening
                await client.get(f"http://127.0.0.1:{port}/mcp", timeout=2.0)
                break
        except Exception:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                raise RuntimeError(f"HTTP server exited early: {stderr}")
            continue
    else:
        proc.kill()
        raise RuntimeError("HTTP server did not start in time")

    try:
        async with streamablehttp_client(url, timeout=600, sse_read_timeout=600) as (read_stream, write_stream, _):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                yield session
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


async def call_diagnostics(session: ClientSession, file_path: str) -> dict:
    """Call lean_diagnostic_messages and return parsed JSON result."""
    result = await session.call_tool(
        "lean_diagnostic_messages",
        {"file_path": file_path},
    )
    text = "\n".join(
        block.text for block in result.content if hasattr(block, "text")
    )
    return json.loads(text)


# ── Test helpers ────────────────────────────────────────────────────────────


def skip_if_no_files():
    if not Path(STUCK_FILE).exists():
        pytest.skip(f"ReproSlow.lean not found at {STUCK_FILE}")
    if not Path(NORMAL_FILE).exists():
        pytest.skip(f"ReproStuck.lean not found at {NORMAL_FILE}")


# ── Tests: WITH FIX (max_wait active on server) ────────────────────────────
# The server.py has max_wait=300 for lean_diagnostic_messages.
# We test that stuck files return (eventually) and normal files return quickly.
# Since 300s is too long for a test, we test at the leanclient layer with 30s.


@pytest.mark.timeout(120)
class TestWithFixLeanclient:
    """Test max_wait at the leanclient layer (bypasses MCP transport)."""

    def test_stuck_file_returns_with_timeout(self):
        skip_if_no_files()
        from leanclient import LeanLSPClient

        client = LeanLSPClient(str(QI_PROJECT))
        try:
            t0 = time.time()
            result = client.get_diagnostics(
                "QuantumInformation/ReproSlow.lean",
                inactivity_timeout=15.0,
                max_wait=MAX_WAIT_FIX,
            )
            elapsed = time.time() - t0

            assert result.timed_out is True, f"Expected timed_out=True, got {result.timed_out}"
            assert result.success is False
            assert elapsed < MAX_WAIT_FIX + 15, f"Took {elapsed:.1f}s, expected ~{MAX_WAIT_FIX}s"
            assert elapsed >= MAX_WAIT_FIX - 1, f"Returned too fast ({elapsed:.1f}s)"
        finally:
            client.close()

    def test_normal_file_unaffected(self):
        skip_if_no_files()
        from leanclient import LeanLSPClient

        client = LeanLSPClient(str(QI_PROJECT))
        try:
            t0 = time.time()
            result = client.get_diagnostics(
                "QuantumInformation/ReproStuck.lean",
                inactivity_timeout=15.0,
                max_wait=120.0,
            )
            elapsed = time.time() - t0

            assert result.timed_out is False, f"Expected timed_out=False, got {result.timed_out}"
            assert result.success is True
            assert elapsed < 60, f"Normal file took too long ({elapsed:.1f}s)"
        finally:
            client.close()


@pytest.mark.timeout(120)
class TestWithoutFixLeanclient:
    """Test that WITHOUT max_wait, the stuck file blocks indefinitely."""

    def test_stuck_file_blocks_without_max_wait(self):
        """Prove the bug: without max_wait, get_diagnostics blocks forever."""
        skip_if_no_files()

        # Run in subprocess with timeout to prove it blocks
        script = f"""
import sys
sys.path.insert(0, "{REPO_ROOT / '.venv/lib/python3.10/site-packages'}")
from leanclient import LeanLSPClient
client = LeanLSPClient("{QI_PROJECT}")
result = client.get_diagnostics("QuantumInformation/ReproSlow.lean", inactivity_timeout=15.0)
client.close()
print("RETURNED")
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, _ = proc.communicate(timeout=NO_FIX_BLOCK_TIMEOUT)
            # If it returned, the bug isn't present (shouldn't happen)
            assert False, f"Expected to block but returned: {stdout.decode()}"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            # Good — it blocked as expected (the bug)

    def test_normal_file_still_works(self):
        """Normal files work fine even without max_wait."""
        skip_if_no_files()
        from leanclient import LeanLSPClient

        client = LeanLSPClient(str(QI_PROJECT))
        try:
            t0 = time.time()
            result = client.get_diagnostics(
                "QuantumInformation/ReproStuck.lean",
                inactivity_timeout=15.0,
                # no max_wait
            )
            elapsed = time.time() - t0

            assert result.success is True
            assert elapsed < 60, f"Normal file took too long ({elapsed:.1f}s)"
        finally:
            client.close()


# ── Tests: Through MCP server (stdio transport) ────────────────────────────


@pytest.mark.timeout(420)
class TestMCPStdio:
    """Test through MCP protocol over stdio transport."""

    @pytest.mark.asyncio
    async def test_normal_file_via_stdio(self):
        skip_if_no_files()
        async with connect_stdio() as session:
            t0 = time.time()
            data = await call_diagnostics(session, NORMAL_FILE)
            elapsed = time.time() - t0

            assert data["timed_out"] is False
            assert data["success"] is True
            assert elapsed < 60

    @pytest.mark.asyncio
    async def test_stuck_file_via_stdio(self):
        """Stuck file returns with timed_out=True via stdio (server max_wait=300)."""
        skip_if_no_files()
        async with connect_stdio() as session:
            t0 = time.time()
            data = await call_diagnostics(session, STUCK_FILE)
            elapsed = time.time() - t0

            assert data["timed_out"] is True, f"Expected timed_out=True, got {data}"
            assert data["success"] is False
            assert elapsed >= 280, f"Returned too fast ({elapsed:.1f}s), expected ~300s"
            assert elapsed < 360, f"Took too long ({elapsed:.1f}s), expected ~300s"


@pytest.mark.timeout(420)
class TestMCPHttp:
    """Test through MCP protocol over streamable-http transport."""

    @pytest.mark.asyncio
    async def test_normal_file_via_http(self):
        skip_if_no_files()
        async with connect_http() as session:
            t0 = time.time()
            data = await call_diagnostics(session, NORMAL_FILE)
            elapsed = time.time() - t0

            assert data["timed_out"] is False
            assert data["success"] is True
            assert elapsed < 60

    @pytest.mark.asyncio
    async def test_stuck_file_via_http(self):
        """Stuck file returns with timed_out=True via HTTP (server max_wait=300)."""
        skip_if_no_files()
        async with connect_http() as session:
            t0 = time.time()
            data = await call_diagnostics(session, STUCK_FILE)
            elapsed = time.time() - t0

            assert data["timed_out"] is True, f"Expected timed_out=True, got {data}"
            assert data["success"] is False
            assert elapsed >= 280, f"Returned too fast ({elapsed:.1f}s), expected ~300s"
            assert elapsed < 360, f"Took too long ({elapsed:.1f}s), expected ~300s"
