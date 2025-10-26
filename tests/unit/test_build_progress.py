"""Unit tests for lean_build progress reporting."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.server import lsp_build


@pytest.mark.asyncio
async def test_lean_build_parses_progress_from_verbose_output():
    """Test that lean_build correctly parses [n/m] progress markers."""

    # Mock verbose lake build output
    mock_build_output = b"""info: test: no previous manifest
\xe2\x9c\x94 [0/8] Ran job computation
\xe2\x9c\x94 [1/8] Ran test:extraDep
\xe2\x84\xb9 [2/8] Built TestProject.Basic (1.6s)
\xe2\x84\xb9 [3/8] Built TestProject (412ms)
\xe2\x84\xb9 [4/8] Built Main (386ms)
\xe2\x84\xb9 [5/8] Built Main:c.o (with exports) (1.3s)
\xe2\x84\xb9 [6/10] Built TestProject.Basic:c.o (with exports) (2.1s)
\xe2\x84\xb9 [7/10] Built TestProject:c.o (with exports) (1.7s)
\xe2\x84\xb9 [8/10] Built test:exe (539ms)
Build completed successfully (8 jobs).
"""

    # Track progress reports
    progress_calls = []

    # Create mock context
    mock_ctx = MagicMock()
    mock_ctx.request_context.lifespan_context.lean_project_path = None
    mock_ctx.request_context.lifespan_context.client = None
    mock_ctx.request_context.lifespan_context.file_content_hashes = {}

    async def track_progress(progress, total, message):
        progress_calls.append({"progress": progress, "total": total, "message": message})

    mock_ctx.report_progress = AsyncMock(side_effect=track_progress)

    # Mock subprocess
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.wait = AsyncMock()

    # Simulate streaming output line by line
    lines = mock_build_output.split(b'\n')
    async def mock_readline():
        if lines:
            return lines.pop(0) + b'\n'
        return b''

    mock_process.stdout.readline = mock_readline

    # Mock asyncio.create_subprocess_exec
    mock_subprocess = AsyncMock(return_value=mock_process)

    # Mock LeanLSPClient
    mock_client = MagicMock()

    with patch('lean_lsp_mcp.server.asyncio.create_subprocess_exec', mock_subprocess), \
         patch('lean_lsp_mcp.server.LeanLSPClient', return_value=mock_client), \
         patch('lean_lsp_mcp.server.OutputCapture'), \
         patch('lean_lsp_mcp.server.subprocess.run'):

        result = await lsp_build(mock_ctx, lean_project_path="/fake/path")

        # Verify progress was reported
        assert len(progress_calls) > 0, "No progress updates were reported"

        # Verify we captured the progress markers
        # We expect to see [0/8], [1/8], [2/8], [3/8], [4/8], [5/8], [6/10], [7/10], [8/10]
        expected_progress = [
            (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 10), (7, 10), (8, 10)
        ]

        actual_progress = [(call["progress"], call["total"]) for call in progress_calls]
        assert actual_progress == expected_progress, \
            f"Expected progress {expected_progress} but got {actual_progress}"

        # Verify dynamic total (changes from 8 to 10)
        totals = [call["total"] for call in progress_calls]
        assert 8 in totals, "Should have seen total=8"
        assert 10 in totals, "Should have seen total=10 (dynamic total update)"

        # Verify messages contain build descriptions
        messages = [call["message"] for call in progress_calls]
        assert any("Built" in msg for msg in messages), \
            "Progress messages should contain 'Built'"


@pytest.mark.asyncio
async def test_lean_build_handles_no_progress_markers():
    """Test that lean_build works even if there are no progress markers."""

    # Mock output without progress markers
    mock_build_output = b"""Building project...
Compiling files...
Done.
"""

    progress_calls = []

    mock_ctx = MagicMock()
    mock_ctx.request_context.lifespan_context.lean_project_path = None
    mock_ctx.request_context.lifespan_context.client = None
    mock_ctx.request_context.lifespan_context.file_content_hashes = {}

    async def track_progress(progress, total, message):
        progress_calls.append({"progress": progress, "total": total, "message": message})

    mock_ctx.report_progress = AsyncMock(side_effect=track_progress)

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.wait = AsyncMock()

    lines = mock_build_output.split(b'\n')
    async def mock_readline():
        if lines:
            return lines.pop(0) + b'\n'
        return b''

    mock_process.stdout.readline = mock_readline

    mock_subprocess = AsyncMock(return_value=mock_process)
    mock_client = MagicMock()

    with patch('lean_lsp_mcp.server.asyncio.create_subprocess_exec', mock_subprocess), \
         patch('lean_lsp_mcp.server.LeanLSPClient', return_value=mock_client), \
         patch('lean_lsp_mcp.server.OutputCapture'), \
         patch('lean_lsp_mcp.server.subprocess.run'):

        result = await lsp_build(mock_ctx, lean_project_path="/fake/path")

        # Should complete without error even with no progress markers
        assert "Error during build" not in result

        # No progress should be reported if there are no markers
        assert len(progress_calls) == 0, \
            "Should not report progress when no markers are present"
