"""Tests for loogle functionality."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.loogle import LoogleManager, get_cache_dir


class TestGetCacheDir:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("LEAN_LOOGLE_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: Path("/home/user"))
        assert get_cache_dir() == Path("/home/user/.cache/lean-lsp-mcp/loogle")

    def test_xdg(self, monkeypatch):
        monkeypatch.delenv("LEAN_LOOGLE_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
        assert get_cache_dir() == Path("/xdg/lean-lsp-mcp/loogle")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LEAN_LOOGLE_CACHE_DIR", "/custom")
        assert get_cache_dir() == Path("/custom")


class TestLoogleManager:
    @pytest.fixture
    def mgr(self, tmp_path):
        return LoogleManager(cache_dir=tmp_path / "loogle")

    def test_binary_path(self, mgr):
        assert mgr.binary_path == mgr.repo_dir / ".lake" / "build" / "bin" / "loogle"

    def test_is_installed(self, mgr):
        assert not mgr.is_installed
        mgr.binary_path.parent.mkdir(parents=True)
        mgr.binary_path.touch()
        assert mgr.is_installed

    @pytest.mark.parametrize(
        "missing,expected_msg", [("git", "git not found"), ("lake", "lake not found")]
    )
    def test_prerequisites_missing(self, mgr, monkeypatch, missing, expected_msg):
        monkeypatch.setattr(
            "shutil.which", lambda c: None if c == missing else f"/bin/{c}"
        )
        ok, msg = mgr._check_prerequisites()
        assert not ok and expected_msg in msg

    def test_prerequisites_ok(self, mgr, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda c: f"/bin/{c}")
        assert mgr._check_prerequisites() == (True, "")

    def test_is_running(self, mgr):
        assert not mgr.is_running
        mgr.process = MagicMock(returncode=None)
        mgr._ready = True
        assert mgr.is_running
        mgr.process.returncode = 1
        assert not mgr.is_running

    def test_clone_repo_exists(self, mgr):
        mgr.repo_dir.mkdir(parents=True)
        assert mgr._clone_repo()

    def test_clone_repo_success(self, mgr):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert mgr._clone_repo()

    def test_clone_repo_fail(self, mgr):
        with patch(
            "subprocess.run", return_value=MagicMock(returncode=1, stderr="err")
        ):
            assert not mgr._clone_repo()

    def test_mathlib_version(self, mgr):
        mgr.repo_dir.mkdir(parents=True)
        (mgr.repo_dir / "lake-manifest.json").write_text(
            json.dumps({"packages": [{"name": "mathlib", "rev": "abc123def456"}]})
        )
        assert mgr._get_mathlib_version() == "abc123def456"
        (mgr.repo_dir / "lake-manifest.json").unlink()
        assert mgr._get_mathlib_version() == "unknown"

    @pytest.mark.asyncio
    async def test_query_not_ready(self, mgr):
        # Without binary installed, start() fails, so query should fail after retry
        with pytest.raises(RuntimeError, match="Failed to start"):
            await mgr.query("test")

    @pytest.mark.asyncio
    async def test_query_success(self, mgr):
        mgr._ready = True
        proc = AsyncMock()
        proc.returncode = None
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdout.readline = AsyncMock(
            return_value=json.dumps(
                {
                    "hits": [
                        {
                            "name": "Nat.add",
                            "type": "Nat → Nat",
                            "module": "Init",
                            "doc": "doc",
                        }
                    ]
                }
            ).encode()
        )
        mgr.process = proc
        r = await mgr.query("Nat", 2)
        assert r == [
            {"name": "Nat.add", "type": "Nat → Nat", "module": "Init", "doc": "doc"}
        ]

    @pytest.mark.asyncio
    async def test_query_error(self, mgr):
        mgr._ready = True
        proc = AsyncMock()
        proc.returncode = None
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdout.readline = AsyncMock(
            return_value=json.dumps({"error": "parse error"}).encode()
        )
        mgr.process = proc
        assert await mgr.query("bad") == []

    @pytest.mark.asyncio
    async def test_query_timeout(self, mgr):
        mgr._ready = True
        proc = AsyncMock()
        proc.returncode = None
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError())
        mgr.process = proc
        with pytest.raises(RuntimeError, match="timeout"):
            await mgr.query("test")

    @pytest.mark.asyncio
    async def test_stop(self, mgr):
        proc = MagicMock()
        proc.returncode = None
        proc.terminate = MagicMock()
        proc.wait = AsyncMock()
        mgr.process, mgr._ready = proc, True
        await mgr.stop()
        proc.terminate.assert_called_once()
        assert mgr.process is None and not mgr._ready

    @pytest.mark.asyncio
    async def test_stop_force_kill(self, mgr):
        proc = MagicMock()
        proc.returncode = None
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        # First wait (after terminate) times out, second wait (after kill) succeeds
        proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), None])
        mgr.process = proc
        await mgr.stop()
        proc.kill.assert_called_once()
        assert proc.wait.await_count == 2

    def test_ensure_installed_no_prereqs(self, tmp_path, monkeypatch):
        mgr = LoogleManager(cache_dir=tmp_path)
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert not mgr.ensure_installed()

    @pytest.mark.asyncio
    async def test_start_not_installed(self, tmp_path):
        assert not await LoogleManager(cache_dir=tmp_path).start()

    def test_cleanup_old_indices(self, mgr):
        mgr.index_dir.mkdir(parents=True)
        # Create some old index files
        (mgr.index_dir / "mathlib-old1.idx").touch()
        (mgr.index_dir / "mathlib-old2.idx").touch()
        current = mgr._get_index_path()
        current.touch()

        mgr._cleanup_old_indices()

        # Only current should remain
        remaining = list(mgr.index_dir.glob("*.idx"))
        assert len(remaining) == 1
        assert remaining[0] == current


@pytest.mark.slow
class TestLoogleIntegration:
    """Integration tests that actually download and run loogle.

    Run with: pytest -m slow tests/unit/test_loogle.py
    These tests require git, lake, and ~2GB disk space.
    Skipped by default in CI.
    """

    @pytest.mark.asyncio
    async def test_local_loogle_full_workflow(self, tmp_path):
        """Test the complete workflow: install, start, query, stop."""
        import shutil

        if not shutil.which("git") or not shutil.which("lake"):
            pytest.skip("git and lake required for integration test")

        mgr = LoogleManager(cache_dir=tmp_path / "loogle")

        try:
            # Install (this takes several minutes on first run)
            assert mgr.ensure_installed(), "Failed to install loogle"
            assert mgr.is_installed

            # Start subprocess
            started = await mgr.start()
            assert started, "Failed to start loogle"
            assert mgr.is_running

            # Query
            results = await mgr.query("Nat.add", num_results=3)
            assert isinstance(results, list)
            assert len(results) > 0
            assert any("add" in r.get("name", "").lower() for r in results)

        finally:
            await mgr.stop()
            assert not mgr.is_running
