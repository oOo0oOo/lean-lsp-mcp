"""Unit tests for local loogle functionality."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lean_lsp_mcp.loogle_local import LoogleManager, get_cache_dir


class TestGetCacheDir:
    """Tests for cache directory resolution."""

    def test_default_cache_dir(self, monkeypatch):
        """Test default cache directory uses ~/.cache."""
        monkeypatch.delenv("LEAN_LOOGLE_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: Path("/home/testuser"))

        result = get_cache_dir()

        assert result == Path("/home/testuser/.cache/lean-lsp-mcp/loogle")

    def test_xdg_cache_dir(self, monkeypatch):
        """Test XDG_CACHE_HOME is respected."""
        monkeypatch.delenv("LEAN_LOOGLE_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")

        result = get_cache_dir()

        assert result == Path("/custom/cache/lean-lsp-mcp/loogle")

    def test_env_override(self, monkeypatch):
        """Test LEAN_LOOGLE_CACHE_DIR overrides all defaults."""
        monkeypatch.setenv("LEAN_LOOGLE_CACHE_DIR", "/my/custom/loogle")

        result = get_cache_dir()

        assert result == Path("/my/custom/loogle")


class TestLoogleManager:
    """Tests for LoogleManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a LoogleManager with a temporary cache directory."""
        return LoogleManager(cache_dir=tmp_path / "loogle")

    def test_binary_path(self, manager):
        """Test binary path is correctly computed."""
        assert manager.binary_path == manager.repo_dir / ".lake" / "build" / "bin" / "loogle"

    def test_is_installed_false_when_no_binary(self, manager):
        """Test is_installed returns False when binary doesn't exist."""
        assert manager.is_installed is False

    def test_is_installed_true_when_binary_exists(self, manager):
        """Test is_installed returns True when binary exists."""
        manager.binary_path.parent.mkdir(parents=True, exist_ok=True)
        manager.binary_path.touch()

        assert manager.is_installed is True

    def test_check_prerequisites_git_missing(self, manager, monkeypatch):
        """Test prerequisites check fails when git is missing."""
        monkeypatch.setattr("shutil.which", lambda cmd: None if cmd == "git" else "/usr/bin/lake")

        ok, msg = manager._check_prerequisites()

        assert ok is False
        assert "git not found" in msg

    def test_check_prerequisites_lake_missing(self, manager, monkeypatch):
        """Test prerequisites check fails when lake is missing."""
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/git" if cmd == "git" else None)

        ok, msg = manager._check_prerequisites()

        assert ok is False
        assert "lake not found" in msg

    def test_check_prerequisites_all_present(self, manager, monkeypatch):
        """Test prerequisites check passes when all tools are present."""
        monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

        ok, msg = manager._check_prerequisites()

        assert ok is True
        assert msg == ""

    def test_get_lake_env_disables_artifact_cache(self, manager):
        """Test _get_lake_env sets LAKE_ARTIFACT_CACHE=false."""
        env = manager._get_lake_env()

        assert env["LAKE_ARTIFACT_CACHE"] == "false"
        # Should inherit other env vars
        assert "PATH" in env

    def test_is_running_false_when_not_started(self, manager):
        """Test is_running returns False when process not started."""
        assert manager.is_running is False

    def test_is_running_true_when_process_running(self, manager):
        """Test is_running returns True when process is running and ready."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is still running
        manager.process = mock_process
        manager._ready = True

        assert manager.is_running is True

    def test_is_running_false_when_process_died(self, manager):
        """Test is_running returns False when process has terminated."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process has exited
        manager.process = mock_process
        manager._ready = True

        assert manager.is_running is False

    def test_clone_repo_already_exists(self, manager):
        """Test clone_repo returns True when repo already exists."""
        manager.repo_dir.mkdir(parents=True, exist_ok=True)

        result = manager._clone_repo()

        assert result is True

    def test_clone_repo_success(self, manager, monkeypatch):
        """Test clone_repo succeeds with mocked subprocess."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = manager._clone_repo()

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "git" in call_args[0][0]
            assert "clone" in call_args[0][0]

    def test_clone_repo_failure(self, manager, monkeypatch):
        """Test clone_repo returns False on failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "clone failed"

        with patch("subprocess.run", return_value=mock_result):
            result = manager._clone_repo()

            assert result is False

    def test_get_mathlib_version_from_manifest(self, manager):
        """Test mathlib version extraction from lake-manifest.json."""
        manager.repo_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "packages": [
                {"name": "mathlib", "rev": "abc123def456"},
                {"name": "other", "rev": "xyz789"},
            ]
        }
        (manager.repo_dir / "lake-manifest.json").write_text(json.dumps(manifest))

        version = manager._get_mathlib_version()

        assert version == "abc123def456"

    def test_get_mathlib_version_missing_manifest(self, manager):
        """Test mathlib version returns 'unknown' when manifest is missing."""
        version = manager._get_mathlib_version()

        assert version == "unknown"

    def test_query_not_ready(self, manager):
        """Test query raises when subprocess is not ready."""
        manager._ready = False
        manager._restart_count = 1  # Prevent restart attempts

        with pytest.raises(RuntimeError, match="not ready"):
            manager.query("test query")

    def test_query_success(self, manager):
        """Test query returns results on success."""
        manager._ready = True
        manager._restart_count = 1  # Set to 1 to verify it gets reset

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.fileno.return_value = 0

        response = json.dumps({
            "hits": [
                {"name": "Nat.add", "type": "(a b : Nat) → Nat", "module": "Init", "doc": "Addition"},
                {"name": "Nat.mul", "type": "(a b : Nat) → Nat", "module": "Init", "doc": None},
            ]
        })
        mock_process.stdout.readline.return_value = response.encode()

        manager.process = mock_process

        with patch("select.select", return_value=([mock_process.stdout], [], [])):
            results = manager.query("Nat", num_results=2)

        assert len(results) == 2
        assert results[0]["name"] == "Nat.add"
        assert results[0]["doc"] == "Addition"
        assert results[1]["name"] == "Nat.mul"
        assert results[1]["doc"] is None
        # Verify restart count was reset
        assert manager._restart_count == 0

    def test_query_empty_hits(self, manager):
        """Test query returns empty list when no hits."""
        manager._ready = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.fileno.return_value = 0

        response = json.dumps({"hits": []})
        mock_process.stdout.readline.return_value = response.encode()

        manager.process = mock_process

        with patch("select.select", return_value=([mock_process.stdout], [], [])):
            results = manager.query("nonexistent")

        assert results == []

    def test_query_error_response(self, manager):
        """Test query returns empty list on error response."""
        manager._ready = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.fileno.return_value = 0

        response = json.dumps({"error": "parse error in query"})
        mock_process.stdout.readline.return_value = response.encode()

        manager.process = mock_process

        with patch("select.select", return_value=([mock_process.stdout], [], [])):
            results = manager.query("bad query")

        assert results == []

    def test_query_timeout(self, manager):
        """Test query raises on timeout."""
        manager._ready = True
        manager._restart_count = 1  # Prevent restart

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.fileno.return_value = 0

        manager.process = mock_process

        with patch("select.select", return_value=([], [], [])):  # Empty = timeout
            with pytest.raises(RuntimeError, match="timeout"):
                manager.query("test")

    def test_stop(self, manager):
        """Test stop terminates the subprocess."""
        mock_process = MagicMock()
        manager.process = mock_process
        manager._ready = True

        manager.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert manager.process is None
        assert manager._ready is False

    def test_stop_force_kill_on_timeout(self, manager):
        """Test stop force kills on wait timeout."""
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("loogle", 5)
        manager.process = mock_process
        manager._ready = True

        manager.stop()

        mock_process.kill.assert_called_once()
        assert manager.process is None


class TestLoogleManagerIntegration:
    """Integration-style tests (still mocked but testing more flow)."""

    def test_ensure_installed_fails_without_prerequisites(self, tmp_path, monkeypatch):
        """Test ensure_installed fails gracefully without prerequisites."""
        manager = LoogleManager(cache_dir=tmp_path)
        monkeypatch.setattr("shutil.which", lambda _: None)

        result = manager.ensure_installed()

        assert result is False

    def test_start_without_installation(self, tmp_path):
        """Test start fails when not installed."""
        manager = LoogleManager(cache_dir=tmp_path)

        result = manager.start()

        assert result is False
