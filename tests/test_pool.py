"""Tests for REPL pool."""

import shutil
from pathlib import Path

import pytest

from lean_lsp_mcp.pool import PoolManager, split_code
from lean_lsp_mcp.pool.settings import PoolSettings


# =============================================================================
# Unit Tests
# =============================================================================


@pytest.mark.parametrize(
    "code,header,body",
    [
        ("import Mathlib\n\ndef x := 1", "import Mathlib", "def x := 1"),
        (
            "import Foo\nimport Bar\n\ndef x := 1",
            "import Foo\nimport Bar",
            "def x := 1",
        ),
        ("def x := 1", "", "def x := 1"),
        ("import Foo\nimport Foo\n\ndef x := 1", "import Foo", "def x := 1"),
        ("import Foo\n\n\ndef x := 1", "import Foo", "def x := 1"),
    ],
)
def test_split_code(code: str, header: str, body: str):
    result = split_code(code)
    assert result.header == header
    assert result.body == body


def test_split_code_mathlib_consolidation():
    code = (
        "import Mathlib.Data.Nat\nimport Mathlib.Data.List\nimport Other\n\ndef x := 1"
    )
    result = split_code(code)
    assert result.header.count("import Mathlib") == 1
    assert "import Other" in result.header


@pytest.mark.parametrize(
    "env,workers,timeout,mem_mb",
    [
        ({}, None, 60, None),  # None means check >= 1 / > 0
        (
            {
                "LEAN_REPL_WORKERS": "4",
                "LEAN_REPL_TIMEOUT": "120",
                "LEAN_REPL_MEM": "4G",
            },
            4,
            120,
            4096,
        ),
        ({"LEAN_REPL_MEM": "512M"}, None, 60, 512),
    ],
)
def test_pool_settings(monkeypatch, env, workers, timeout, mem_mb):
    for k in [
        "LEAN_REPL_WORKERS",
        "LEAN_REPL_TIMEOUT",
        "LEAN_REPL_MEM",
        "LEAN_REPL_PATH",
    ]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    settings = PoolSettings.from_env()

    if workers is None:
        assert settings.workers >= 1
    else:
        assert settings.workers == workers

    assert settings.timeout == timeout

    if mem_mb is None:
        assert settings.mem_mb > 0
    else:
        assert settings.mem_mb == mem_mb


# =============================================================================
# Integration Tests (require REPL binary)
# =============================================================================


@pytest.fixture
async def pool(test_project_path: Path, monkeypatch):
    # Check multiple possible locations for REPL binary
    candidates = [
        test_project_path / ".lake" / "build" / "bin" / "repl",
        test_project_path
        / ".lake"
        / "packages"
        / "repl"
        / ".lake"
        / "build"
        / "bin"
        / "repl",
    ]
    repl = next((p for p in candidates if p.exists()), None)
    if not repl:
        found = shutil.which("repl")
        repl = Path(found) if found else None
    if not repl:
        pytest.skip("REPL binary not found")

    monkeypatch.setenv("LEAN_REPL_PATH", str(repl))
    monkeypatch.setenv("LEAN_REPL_WORKERS", "2")
    monkeypatch.setenv("LEAN_REPL_TIMEOUT", "30")

    manager = PoolManager(project_dir=str(test_project_path))
    yield manager
    await manager.cleanup()


@pytest.mark.asyncio
async def test_multi_attempt_returns_results(pool: PoolManager):
    results = await pool.run_multi_attempt("theorem t : 1 = 1 := by\n", ["rfl", "simp"])
    assert len(results) == 2
    assert any(r.error is None for r in results)


@pytest.mark.asyncio
async def test_header_caching_reuses_worker(pool: PoolManager):
    base = "theorem t : 1 = 1 := by\n"

    await pool.run_multi_attempt(base, ["rfl"])
    assert len(pool._free) == 1

    await pool.run_multi_attempt(base, ["rfl"])
    assert len(pool._free) == 1  # Still 1, reused


@pytest.mark.asyncio
async def test_backtracking_isolation(pool: PoolManager):
    base = "theorem t : 1 + 1 = 2 := by\n"
    results = await pool.run_multi_attempt(
        base,
        [
            "have h : False := sorry; exact h.elim",
            "rfl",
        ],
    )
    assert len(results) == 2
    assert results[1].error is None  # rfl specifically should work
