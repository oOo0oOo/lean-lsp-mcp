"""Unit tests for lean_build."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.server import lsp_build


@pytest.fixture
def build_mocks():
    """Shared mocks for lsp_build tests."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context.lean_project_path = None
    ctx.request_context.lifespan_context.client = None
    ctx.report_progress = AsyncMock()

    # Simple process for cache (no stdout needed)
    cache_proc = MagicMock()
    cache_proc.wait = AsyncMock()

    # Build process with stdout
    build_proc = MagicMock()
    build_proc.returncode = 0
    build_proc.wait = AsyncMock()

    return ctx, cache_proc, build_proc


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
        patch("lean_lsp_mcp.server.LeanLSPClient"),
        patch("lean_lsp_mcp.server.OutputCapture"),
    ):
        yield mock_exec


@pytest.mark.asyncio
async def test_progress_parsing(build_mocks, patch_build):
    """Progress markers [n/m] are parsed and reported."""
    ctx, cache_proc, build_proc = build_mocks
    progress_calls = []
    ctx.report_progress = AsyncMock(
        side_effect=lambda progress, total, message: progress_calls.append(
            (progress, total, message)
        )
    )

    build_proc.stdout.readline = make_readline(
        b"[0/8] Ran job\n[1/8] Built A\n[2/10] Built B\n"
    )
    patch_build.side_effect = [cache_proc, build_proc]

    await lsp_build(ctx, lean_project_path="/fake")

    # Check build progress calls (exclude setup phases)
    build_progress = [
        (p, t) for p, t, m in progress_calls if "Built" in m or "Ran" in m
    ]
    assert build_progress == [(0, 8), (1, 8), (2, 10)]


@pytest.mark.asyncio
async def test_filters_trace_lines(build_mocks, patch_build):
    """Verbose trace: and LEAN_PATH= lines are filtered from output."""
    ctx, cache_proc, build_proc = build_mocks
    build_proc.stdout.readline = make_readline(
        b"[0/2] Built A\ntrace: .> LEAN_PATH=/x lean cmd\n[1/2] Built B\n"
    )
    patch_build.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=100)

    output_text = "\n".join(result.output)
    assert "trace:" not in output_text
    assert "LEAN_PATH" not in output_text
    assert "Built" in output_text


@pytest.mark.asyncio
async def test_output_truncation(build_mocks, patch_build):
    """output_lines parameter truncates to last N lines."""
    ctx, cache_proc, build_proc = build_mocks
    lines = b"\n".join(f"[{i}/50] Built M{i}".encode() for i in range(50))
    build_proc.stdout.readline = make_readline(lines + b"\nDone\n")
    patch_build.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=5)

    # Should only have last 5 lines
    assert len(result.output) <= 5


@pytest.mark.asyncio
async def test_output_lines_zero(build_mocks, patch_build):
    """output_lines=0 returns empty output."""
    ctx, cache_proc, build_proc = build_mocks
    build_proc.stdout.readline = make_readline(b"[0/1] Built\nDone\n")
    patch_build.side_effect = [cache_proc, build_proc]

    result = await lsp_build(ctx, lean_project_path="/fake", output_lines=0)

    assert result.output == []
    assert result.success


@pytest.mark.asyncio
async def test_reports_cache_progress(build_mocks, patch_build):
    """Cache fetch is reported via progress."""
    ctx, cache_proc, build_proc = build_mocks
    progress_calls = []
    ctx.report_progress = AsyncMock(
        side_effect=lambda progress, total, message: progress_calls.append(
            (progress, total, message)
        )
    )
    build_proc.stdout.readline = make_readline(b"Done\n")
    patch_build.side_effect = [cache_proc, build_proc]

    await lsp_build(ctx, lean_project_path="/fake", output_lines=100)

    # Should have reported cache fetch progress
    assert any("cache" in m.lower() for p, t, m in progress_calls)
