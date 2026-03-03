"""Unit tests for lean_build."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.server import lsp_build


class _FailingClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise PermissionError("operation not permitted")


@pytest.fixture
def build_mocks():
    """Shared mocks for lsp_build tests."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context.lean_project_path = None
    ctx.request_context.lifespan_context.client = None
    ctx.report_progress = AsyncMock()

    # Simple process for clean/cache (no stdout needed)
    clean_proc = MagicMock()
    clean_proc.wait = AsyncMock()

    cache_proc = MagicMock()
    cache_proc.wait = AsyncMock()

    # Build process with stdout
    build_proc = MagicMock()
    build_proc.returncode = 0
    build_proc.wait = AsyncMock()

    return ctx, clean_proc, cache_proc, build_proc


def make_readline(output: bytes):
    """Create async readline that streams output line by line."""
    lines = output.split(b"\n")

    async def readline():
        return lines.pop(0) + b"\n" if lines else b""

    return readline


@pytest.fixture
def patch_build():
    """Context manager to patch all build dependencies."""
    with (
        patch("lean_lsp_mcp.server.asyncio.create_subprocess_exec") as mock_exec,
        patch("lean_lsp_mcp.server.startup_client") as mock_startup,
    ):
        mock_startup.side_effect = (
            lambda ctx, prevent_cache_get_override=None: setattr(  # pragma: no cover
                ctx.request_context.lifespan_context, "client", MagicMock()
            )
        )
        yield mock_exec, mock_startup


@pytest.mark.asyncio
async def test_progress_parsing(build_mocks, patch_build):
    """Progress markers [n/m] are parsed and reported."""
    ctx, _, cache_proc, build_proc = build_mocks
    progress_calls = []
    ctx.report_progress = AsyncMock(
        side_effect=lambda progress, total, message: progress_calls.append(
            (progress, total, message)
        )
    )

    build_proc.stdout.readline = make_readline(
        b"[0/8] Ran job\n[1/8] Built A\n[2/10] Built B\n"
    )
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    await lsp_build(ctx, lean_project_path="/fake")

    # Check build progress calls (exclude setup phases)
    build_progress = [
        (p, t) for p, t, m in progress_calls if "Built" in m or "Ran" in m
    ]
    assert build_progress == [(0, 8), (1, 8), (2, 10)]


@pytest.mark.asyncio
async def test_filters_trace_lines(build_mocks, patch_build):
    """Verbose trace: and LEAN_PATH= lines are filtered from output."""
    ctx, _, cache_proc, build_proc = build_mocks
    build_proc.stdout.readline = make_readline(
        b"[0/2] Built A\ntrace: .> LEAN_PATH=/x lean cmd\n[1/2] Built B\n"
    )
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=100)

    assert "trace:" not in result.output
    assert "LEAN_PATH" not in result.output
    assert "Built" in result.output


@pytest.mark.asyncio
async def test_output_truncation(build_mocks, patch_build):
    """output_lines parameter truncates to last N lines."""
    ctx, _, cache_proc, build_proc = build_mocks
    lines = b"\n".join(f"[{i}/50] Built M{i}".encode() for i in range(50))
    build_proc.stdout.readline = make_readline(lines + b"\nDone\n")
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=5)

    # Should only have last 5 lines
    assert len(result.output.strip().split("\n")) <= 5


@pytest.mark.asyncio
async def test_output_lines_zero(build_mocks, patch_build):
    """output_lines=0 returns empty output."""
    ctx, _, cache_proc, build_proc = build_mocks
    build_proc.stdout.readline = make_readline(b"[0/1] Built\nDone\n")
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=0)

    assert result.output == ""
    assert result.success


@pytest.mark.asyncio
async def test_reports_cache_progress(build_mocks, patch_build):
    """Cache fetch is reported via progress."""
    ctx, _, cache_proc, build_proc = build_mocks
    progress_calls = []
    ctx.report_progress = AsyncMock(
        side_effect=lambda progress, total, message: progress_calls.append(
            (progress, total, message)
        )
    )
    build_proc.stdout.readline = make_readline(b"Done\n")
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    await lsp_build(ctx, lean_project_path="/fake", output_lines=100)

    # Should have reported cache fetch progress
    assert any("cache" in m.lower() for p, t, m in progress_calls)


@pytest.mark.asyncio
async def test_lsp_build_fails_closed_when_client_close_fails(build_mocks, patch_build):
    """Pre-build client close failure aborts build to avoid stale client reuse."""
    ctx, _, cache_proc, build_proc = build_mocks
    failing_client = _FailingClient()
    ctx.request_context.lifespan_context.client = failing_client

    build_proc.stdout.readline = make_readline(b"[0/1] Built\nDone\n")
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=100)

    assert failing_client.close_calls == 1
    assert not result.success
    assert "restart safety could not be verified" in result.errors[0]
    assert ctx.request_context.lifespan_context.client is failing_client


@pytest.mark.asyncio
async def test_lsp_build_holds_lease_during_restart(build_mocks, patch_build):
    """lean_build must keep an acquired lease reserved until restart completes."""
    ctx, _, cache_proc, build_proc = build_mocks
    old_client = MagicMock()
    coordination_client = MagicMock()

    ctx.request_context.lifespan_context.client = old_client
    ctx.request_context.lifespan_context.coordination_client = coordination_client
    ctx.request_context.lifespan_context.instance_id = "inst-1"
    ctx.request_context.lifespan_context.client_lease_id = "lease-1"
    ctx.request_context.lifespan_context.client_worker_key = "worker-1"

    build_proc.stdout.readline = make_readline(b"[0/1] Built\nDone\n")
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=50)

    assert result.success
    old_client.close.assert_called_once()
    coordination_client.release_lease.assert_not_called()
    assert ctx.request_context.lifespan_context.client_lease_id == "lease-1"
    assert ctx.request_context.lifespan_context.client_worker_key == "worker-1"


@pytest.mark.asyncio
async def test_lsp_build_releases_reserved_lease_on_build_failure(
    build_mocks, patch_build
):
    """If restart fails after reserving a lease, release it to avoid stale capacity usage."""
    ctx, _, cache_proc, build_proc = build_mocks
    old_client = MagicMock()
    coordination_client = MagicMock()

    ctx.request_context.lifespan_context.client = old_client
    ctx.request_context.lifespan_context.coordination_client = coordination_client
    ctx.request_context.lifespan_context.instance_id = "inst-1"
    ctx.request_context.lifespan_context.client_lease_id = "lease-1"
    ctx.request_context.lifespan_context.client_worker_key = "worker-1"

    build_proc.returncode = 1
    build_proc.stdout.readline = make_readline(b"[0/1] Building\nerror: boom\n")
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=50)

    assert not result.success
    old_client.close.assert_called_once()
    coordination_client.release_lease.assert_called_once_with(
        instance_id="inst-1",
        lease_id="lease-1",
    )
    assert ctx.request_context.lifespan_context.client_lease_id is None
    assert ctx.request_context.lifespan_context.client_worker_key is None


@pytest.mark.asyncio
async def test_lsp_build_suppresses_clean_and_cache_stdio(
    build_mocks, patch_build, monkeypatch
):
    """clean/cache subprocesses must never inherit server stdio."""
    monkeypatch.setenv("LEAN_LSP_MCP_ACTIVE_TRANSPORT", "stdio")
    ctx, clean_proc, cache_proc, build_proc = build_mocks
    build_proc.stdout.readline = make_readline(b"[0/1] Built\nDone\n")
    mock_exec, _ = patch_build
    mock_exec.side_effect = [clean_proc, cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", clean=True)

    assert result.success

    clean_call = mock_exec.call_args_list[0]
    assert clean_call.kwargs["stdout"] == asyncio.subprocess.DEVNULL
    assert clean_call.kwargs["stderr"] == asyncio.subprocess.DEVNULL

    cache_call = mock_exec.call_args_list[1]
    assert cache_call.kwargs["stdout"] == asyncio.subprocess.DEVNULL
    assert cache_call.kwargs["stderr"] == asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_lsp_build_continues_when_progress_reporting_fails(build_mocks, patch_build):
    """Transport/progress write errors are swallowed for build resiliency."""
    ctx, _, cache_proc, build_proc = build_mocks
    ctx.report_progress = AsyncMock(side_effect=BrokenPipeError("transport closed"))
    build_proc.stdout.readline = make_readline(
        b"[0/2] Built A\n[1/2] Built B\nDone\n"
    )
    mock_exec, _ = patch_build
    mock_exec.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=20)

    assert result.success
