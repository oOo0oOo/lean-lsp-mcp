"""Tests for local loogle functionality."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lean_lsp_mcp.loogle_local import LoogleManager, get_cache_dir


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

    @pytest.mark.parametrize("missing,expected_msg", [("git", "git not found"), ("lake", "lake not found")])
    def test_prerequisites_missing(self, mgr, monkeypatch, missing, expected_msg):
        monkeypatch.setattr("shutil.which", lambda c: None if c == missing else f"/bin/{c}")
        ok, msg = mgr._check_prerequisites()
        assert not ok and expected_msg in msg

    def test_prerequisites_ok(self, mgr, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda c: f"/bin/{c}")
        assert mgr._check_prerequisites() == (True, "")

    def test_lake_env(self, mgr):
        assert mgr._get_lake_env()["LAKE_ARTIFACT_CACHE"] == "false"

    def test_is_running(self, mgr):
        assert not mgr.is_running
        mgr.process = MagicMock(poll=lambda: None)
        mgr._ready = True
        assert mgr.is_running
        mgr.process.poll = lambda: 1
        assert not mgr.is_running

    def test_clone_repo_exists(self, mgr):
        mgr.repo_dir.mkdir(parents=True)
        assert mgr._clone_repo()

    def test_clone_repo_success(self, mgr):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert mgr._clone_repo()

    def test_clone_repo_fail(self, mgr):
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="err")):
            assert not mgr._clone_repo()

    def test_mathlib_version(self, mgr):
        mgr.repo_dir.mkdir(parents=True)
        (mgr.repo_dir / "lake-manifest.json").write_text(
            json.dumps({"packages": [{"name": "mathlib", "rev": "abc123def456"}]})
        )
        assert mgr._get_mathlib_version() == "abc123def456"
        (mgr.repo_dir / "lake-manifest.json").unlink()
        assert mgr._get_mathlib_version() == "unknown"

    def test_query_not_ready(self, mgr):
        mgr._restart_count = 1
        with pytest.raises(RuntimeError, match="not ready"):
            mgr.query("test")

    def test_query_success(self, mgr):
        mgr._ready = True
        mgr._restart_count = 1
        proc = MagicMock(poll=lambda: None, stdin=MagicMock(), stdout=MagicMock(fileno=lambda: 0))
        proc.stdout.readline.return_value = json.dumps(
            {"hits": [{"name": "Nat.add", "type": "Nat → Nat", "module": "Init", "doc": "doc"}]}
        ).encode()
        mgr.process = proc
        with patch("select.select", return_value=([proc.stdout], [], [])):
            r = mgr.query("Nat", 2)
        assert r == [{"name": "Nat.add", "type": "Nat → Nat", "module": "Init", "doc": "doc"}]
        assert mgr._restart_count == 0

    def test_query_error(self, mgr):
        mgr._ready = True
        proc = MagicMock(poll=lambda: None, stdin=MagicMock(), stdout=MagicMock(fileno=lambda: 0))
        proc.stdout.readline.return_value = json.dumps({"error": "parse error"}).encode()
        mgr.process = proc
        with patch("select.select", return_value=([proc.stdout], [], [])):
            assert mgr.query("bad") == []

    def test_query_timeout(self, mgr):
        mgr._ready = True
        mgr._restart_count = 1
        mgr.process = MagicMock(poll=lambda: None, stdin=MagicMock(), stdout=MagicMock(fileno=lambda: 0))
        with patch("select.select", return_value=([], [], [])):
            with pytest.raises(RuntimeError, match="timeout"):
                mgr.query("test")

    def test_stop(self, mgr):
        proc = MagicMock()
        mgr.process, mgr._ready = proc, True
        mgr.stop()
        proc.terminate.assert_called_once()
        assert mgr.process is None and not mgr._ready

    def test_stop_force_kill(self, mgr):
        proc = MagicMock()
        proc.wait.side_effect = subprocess.TimeoutExpired("loogle", 5)
        mgr.process = proc
        mgr.stop()
        proc.kill.assert_called_once()

    def test_ensure_installed_no_prereqs(self, tmp_path, monkeypatch):
        mgr = LoogleManager(cache_dir=tmp_path)
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert not mgr.ensure_installed()

    def test_start_not_installed(self, tmp_path):
        assert not LoogleManager(cache_dir=tmp_path).start()
