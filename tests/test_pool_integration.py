"""Integration tests for REPL pool (requires REPL binary)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lean_lsp_mcp.pool import PoolManager


def _find_repl(project: Path) -> Path | None:
    local = project / ".lake" / "build" / "bin" / "repl"
    if local.exists():
        return local
    found = shutil.which("repl")
    return Path(found) if found else None


@pytest.fixture
def repl_env(test_project_path: Path, monkeypatch):
    repl = _find_repl(test_project_path)
    if not repl:
        pytest.skip("REPL binary not found")
    monkeypatch.setenv("LEAN_REPL_PATH", str(repl))
    monkeypatch.setenv("LEAN_REPL_WORKERS", "1")
    monkeypatch.setenv("LEAN_REPL_TIMEOUT", "30")
    return test_project_path


@pytest.mark.asyncio
async def test_pool_manager_lifecycle(repl_env: Path) -> None:
    manager = PoolManager(project_dir=str(repl_env))
    assert manager.settings.workers == 1
    await manager.cleanup()


@pytest.mark.asyncio
async def test_pool_multi_attempt(repl_env: Path) -> None:
    manager = PoolManager(project_dir=str(repl_env))
    try:
        base_code = "theorem test : 1 + 1 = 2 := by\n"
        results = await manager.run_multi_attempt(base_code, ["rfl", "simp", "decide"])
        assert len(results) == 3
        assert any(r.error is None for r in results)
    finally:
        await manager.cleanup()
