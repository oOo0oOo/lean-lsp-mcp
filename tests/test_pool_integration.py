"""Integration tests for REPL pool manager.

These tests require a Lean project with REPL available.
They are skipped if the REPL binary is not found.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _find_repl_binary(project_path: Path) -> Path | None:
    """Find the REPL binary in the project or globally."""
    # Check project-local REPL
    local_repl = project_path / ".lake" / "build" / "bin" / "repl"
    if local_repl.exists():
        return local_repl

    # Check if 'repl' is in PATH
    if shutil.which("repl"):
        return Path(shutil.which("repl"))  # type: ignore

    return None


@pytest.fixture
def repl_enabled_env(test_project_path: Path) -> dict[str, str]:
    """Environment with REPL pool enabled."""
    repl_binary = _find_repl_binary(test_project_path)
    if repl_binary is None:
        pytest.skip("REPL binary not found; install via 'lake build repl' in test project")

    return {
        "LEAN_REPL": "1",
        "LEAN_REPL_PATH": str(repl_binary),
        "LEAN_PROJECT_PATH": str(test_project_path),
        "LEAN_REPL_WORKERS": "2",
        "LEAN_REPL_TIMEOUT": "30",
    }


@pytest.mark.asyncio
@pytest.mark.slow
async def test_pool_multi_attempt(
    repl_enabled_env: dict[str, str],
    test_project_path: Path,
) -> None:
    """Test multi_attempt with REPL pool."""
    pytest.skip("REPL pool integration test not yet implemented - needs MCP client with pool env")
    # TODO: Implement once we have a way to pass custom env to mcp_client_factory


@pytest.mark.asyncio
async def test_pool_manager_lifecycle(
    repl_enabled_env: dict[str, str],
    test_project_path: Path,
) -> None:
    """Test pool manager can start and stop cleanly."""
    from lean_lsp_mcp.pool import Manager as PoolManager

    repl_path = repl_enabled_env["LEAN_REPL_PATH"]
    project_dir = str(test_project_path)

    # Create manager
    manager = PoolManager(
        repl_path=repl_path,
        project_dir=project_dir,
        max_repls=2,
        command_timeout=30,
    )

    # Check settings were applied
    assert manager.max_repls == 2
    assert manager.command_timeout == 30

    # Clean up should work even before any REPLs started
    await manager.cleanup()


@pytest.mark.asyncio
async def test_pool_manager_run_multi_attempt(
    repl_enabled_env: dict[str, str],
    test_project_path: Path,
) -> None:
    """Test running multi_attempt through pool manager."""
    from lean_lsp_mcp.pool import Manager as PoolManager, NoAvailableReplError

    repl_path = repl_enabled_env["LEAN_REPL_PATH"]
    project_dir = str(test_project_path)

    manager = PoolManager(
        repl_path=repl_path,
        project_dir=project_dir,
        max_repls=1,
        command_timeout=60,
    )

    try:
        # Run a simple multi_attempt
        base_code = """
import Mathlib.Data.Nat.Basic

theorem test_theorem : 1 + 1 = 2 := by
"""
        snippets = ["rfl", "simp", "decide"]

        results = await manager.run_multi_attempt(
            base_code=base_code,
            snippets=snippets,
            timeout=60,
        )

        # Should have one result per snippet
        assert len(results) == 3

        # At least one should succeed (rfl works for 1+1=2)
        success_count = sum(1 for r in results if r.error is None)
        assert success_count >= 1, f"Expected at least one success, got results: {results}"

    finally:
        await manager.cleanup()
