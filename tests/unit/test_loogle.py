"""Tests for loogle functionality."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.loogle import LoogleManager, get_cache_dir


class TestGetCacheDir:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("LEAN_LOOGLE_CACHE_DIR", raising=False)
        if os.name == "nt":
            monkeypatch.delenv("LOCALAPPDATA", raising=False)
            monkeypatch.setattr(Path, "home", lambda: Path("C:/Users/user"))
            assert get_cache_dir() == Path(
                "C:/Users/user/AppData/Local/lean-lsp-mcp/loogle"
            )
        else:
            monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
            monkeypatch.setattr(Path, "home", lambda: Path("/home/user"))
            assert get_cache_dir() == Path("/home/user/.cache/lean-lsp-mcp/loogle")

    def test_xdg(self, monkeypatch):
        monkeypatch.delenv("LEAN_LOOGLE_CACHE_DIR", raising=False)
        if os.name == "nt":
            monkeypatch.setenv("LOCALAPPDATA", "C:/LocalApp")
            assert get_cache_dir() == Path("C:/LocalApp/lean-lsp-mcp/loogle")
        else:
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

    def test_toolchain_version(self, mgr):
        mgr.repo_dir.mkdir(parents=True)
        (mgr.repo_dir / "lean-toolchain").write_text("leanprover/lean4:v4.25.0-rc1")
        assert mgr._get_toolchain_version() == "leanprover/lean4:v4.25.0-rc1"
        (mgr.repo_dir / "lean-toolchain").unlink()
        assert mgr._get_toolchain_version() is None

    def test_check_toolchain_installed(self, mgr, tmp_path, monkeypatch):
        # No lean-toolchain file => OK
        mgr.repo_dir.mkdir(parents=True)
        ok, _ = mgr._check_toolchain_installed()
        assert ok

        # Create a toolchain file for a non-existent version
        (mgr.repo_dir / "lean-toolchain").write_text("leanprover/lean4:v9.9.9")
        # Point to fake elan home
        monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))
        ok, msg = mgr._check_toolchain_installed()
        assert not ok
        assert "v9.9.9" in msg

        # Create the toolchain directory
        tc_dir = tmp_path / "elan" / "toolchains" / "leanprover--lean4---v9.9.9"
        tc_dir.mkdir(parents=True)
        ok, _ = mgr._check_toolchain_installed()
        assert ok

    def test_check_environment(self, mgr, tmp_path, monkeypatch):
        # No binary => not OK
        ok, msg = mgr.check_environment()
        assert not ok
        assert "binary not found" in msg

        # Create binary
        mgr.binary_path.parent.mkdir(parents=True)
        mgr.binary_path.touch()
        mgr.repo_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))

        # No toolchain file => OK
        ok, _ = mgr.check_environment()
        assert ok

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
        from lean_lsp_mcp.loogle import LoogleQueryError

        with pytest.raises(LoogleQueryError, match="parse error"):
            await mgr.query("bad")

    @pytest.mark.asyncio
    async def test_query_timeout(self, mgr):
        mgr._ready = True
        proc = AsyncMock()
        proc.returncode = None
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.kill = MagicMock()  # sync method on the real process
        mgr.process = proc
        with pytest.raises(RuntimeError, match="timeout"):
            await mgr.query("test")
        # Stream is desynced after a timeout: subprocess must be discarded.
        assert mgr.process is None
        assert mgr._ready is False

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

    def test_ensure_installed_handles_cache_permission_error(
        self, tmp_path, monkeypatch
    ):
        mgr = LoogleManager(cache_dir=tmp_path / "loogle")
        monkeypatch.setattr(mgr, "_check_prerequisites", lambda: (True, ""))
        orig_mkdir = Path.mkdir

        def fail_cache_dir(path, *args, **kwargs):
            if path == mgr.cache_dir:
                raise PermissionError("denied")
            return orig_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_cache_dir)
        assert not mgr.ensure_installed()

    @pytest.mark.asyncio
    async def test_start_not_installed(self, tmp_path):
        assert not await LoogleManager(cache_dir=tmp_path).start()

    def test_cleanup_old_indices(self, mgr):
        mgr.index_dir.mkdir(parents=True)
        # Create some old mathlib index files
        (mgr.index_dir / "mathlib-old1.idx").touch()
        (mgr.index_dir / "mathlib-old2.idx").touch()
        # Also create old project-specific indexes
        (mgr.index_dir / "mathlib-old1-abc123.idx").touch()
        (mgr.index_dir / "mathlib-old2-def456.idx").touch()
        current = mgr._get_index_path()
        current.touch()
        # Create a project-specific index for current mathlib version (should be preserved)
        current_mathlib_version = mgr._get_mathlib_version()
        project_index = mgr.index_dir / f"mathlib-{current_mathlib_version}-proj123.idx"
        project_index.touch()

        mgr._cleanup_old_indices()

        # Current and project-specific index for current mathlib version should remain
        remaining = list(mgr.index_dir.glob("*.idx"))
        assert len(remaining) == 2
        assert current in remaining
        assert project_index in remaining

    def test_discover_project_paths_no_project(self, mgr):
        assert mgr._discover_project_paths() == []

    def test_get_project_toolchain(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        mgr = LoogleManager(cache_dir=tmp_path / "cache", project_path=project)
        assert mgr._get_project_toolchain() is None
        (project / "lean-toolchain").write_text("leanprover/lean4:v4.28.0\n")
        assert mgr._get_project_toolchain() == "leanprover/lean4:v4.28.0"

    def test_get_project_toolchain_no_project(self, mgr):
        assert mgr._get_project_toolchain() is None

    def test_discover_project_paths_tc_mismatch(self, tmp_path):
        project = tmp_path / "project"
        project_lib = project / ".lake" / "build" / "lib" / "lean"
        project_lib.mkdir(parents=True)
        (project / "lean-toolchain").write_text("leanprover/lean4:v4.28.0")

        mgr = LoogleManager(cache_dir=tmp_path / "cache", project_path=project)
        mgr.repo_dir.mkdir(parents=True)
        (mgr.repo_dir / "lean-toolchain").write_text("leanprover/lean4:v4.29.0-rc4")

        assert mgr._discover_project_paths() == []

    def test_discover_project_paths_tc_match(self, tmp_path):
        project = tmp_path / "project"
        project_lib = project / ".lake" / "build" / "lib" / "lean"
        project_lib.mkdir(parents=True)
        (project / "lean-toolchain").write_text("leanprover/lean4:v4.28.0")

        mgr = LoogleManager(cache_dir=tmp_path / "cache", project_path=project)
        mgr.repo_dir.mkdir(parents=True)
        (mgr.repo_dir / "lean-toolchain").write_text("leanprover/lean4:v4.28.0")

        assert len(mgr._discover_project_paths()) == 1

    def test_discover_project_paths(self, tmp_path):
        # Create a fake project with packages
        project = tmp_path / "project"
        pkg1_lib = (
            project / ".lake" / "packages" / "pkg1" / ".lake" / "build" / "lib" / "lean"
        )
        pkg2_lib = (
            project / ".lake" / "packages" / "pkg2" / ".lake" / "build" / "lib" / "lean"
        )
        project_lib = project / ".lake" / "build" / "lib" / "lean"
        pkg1_lib.mkdir(parents=True)
        pkg2_lib.mkdir(parents=True)
        project_lib.mkdir(parents=True)

        mgr = LoogleManager(cache_dir=tmp_path / "cache", project_path=project)
        paths = mgr._discover_project_paths()

        assert len(paths) == 3
        assert pkg1_lib in paths
        assert pkg2_lib in paths
        assert project_lib in paths

    def test_index_path_with_project(self, tmp_path):
        # Without project - base index name
        mgr1 = LoogleManager(cache_dir=tmp_path / "cache1")
        path1 = mgr1._get_index_path()
        assert "mathlib-" in path1.name
        assert path1.name.count("-") == 1  # Just mathlib-<version>.idx

        # With extra paths - includes hash
        mgr2 = LoogleManager(cache_dir=tmp_path / "cache2")
        mgr2._extra_paths = [Path("/some/path")]
        path2 = mgr2._get_index_path()
        assert path2.name.count("-") == 2  # mathlib-<version>-<hash>.idx

    def test_set_project_path(self, tmp_path):
        project = tmp_path / "project"
        lib = project / ".lake" / "build" / "lib" / "lean"
        lib.mkdir(parents=True)

        mgr = LoogleManager(cache_dir=tmp_path / "cache")
        assert mgr._extra_paths == []

        # Setting project path discovers paths
        changed = mgr.set_project_path(project)
        assert changed
        assert len(mgr._extra_paths) == 1

        # Setting same path again - no change
        changed = mgr.set_project_path(project)
        assert not changed

    # --- issue #193: search-path and CLI-flag handling ---

    def test_loogle_env_no_extra_paths(self, mgr):
        """Without project paths, LEAN_PATH is left unset (loogle uses its own)."""
        env = mgr._loogle_env()
        assert "LEAN_PATH" not in env
        assert env["LAKE_ARTIFACT_CACHE"] == "false"

    def test_loogle_env_with_extra_paths(self, mgr, monkeypatch):
        """Project paths extend (not replace) loogle's base search path via LEAN_PATH."""
        monkeypatch.setattr(mgr, "_get_base_lean_path", lambda: "/base/a:/base/b")
        mgr._extra_paths = [Path("/proj/p1"), Path("/proj/p2")]
        entries = mgr._loogle_env()["LEAN_PATH"].split(os.pathsep)
        # Loogle's base (toolchain core with Init.olean + mathlib) comes first.
        assert entries[0] == "/base/a"
        assert entries[1] == "/base/b"
        assert str(Path("/proj/p1")) in entries
        assert str(Path("/proj/p2")) in entries

    def test_get_base_lean_path_caches(self, mgr, monkeypatch):
        calls = []

        def fake_run(cmd, timeout=300, cwd=None, env=None):
            calls.append(cmd)
            return MagicMock(returncode=0, stdout="/x/lib:/y/lib\n")

        monkeypatch.setattr(mgr, "_run", fake_run)
        assert mgr._get_base_lean_path() == "/x/lib:/y/lib"
        assert mgr._get_base_lean_path() == "/x/lib:/y/lib"  # cached, no second call
        assert len(calls) == 1
        assert calls[0][:3] == ["lake", "env", "printenv"]

    def test_build_index_uses_index_mode_not_write_index(self, mgr, monkeypatch):
        """Index build uses the new --index-mode/--index-file flags, never --path."""
        mgr.binary_path.parent.mkdir(parents=True)
        mgr.binary_path.touch()
        mgr._extra_paths = [Path("/proj/p1")]
        monkeypatch.setattr(mgr, "_get_base_lean_path", lambda: "/base")
        captured = {}

        def fake_run(cmd, timeout=300, cwd=None, env=None):
            captured["cmd"], captured["env"] = cmd, env
            Path(cmd[cmd.index("--index-file") + 1]).touch()
            return MagicMock(returncode=0)

        monkeypatch.setattr(mgr, "_run", fake_run)
        assert mgr._build_index() is not None
        cmd = captured["cmd"]
        assert "--index-mode" in cmd and "write" in cmd and "--index-file" in cmd
        assert "--write-index" not in cmd
        assert "--path" not in cmd
        assert "LEAN_PATH" in captured["env"]

    @pytest.mark.asyncio
    async def test_start_uses_index_mode_and_lean_path(self, mgr, monkeypatch):
        """start() uses --index-mode/--index-file and LEAN_PATH, never --read-index/--path."""
        mgr.binary_path.parent.mkdir(parents=True)
        mgr.binary_path.touch()
        mgr._extra_paths = [Path("/proj/p1")]
        monkeypatch.setattr(mgr, "check_environment", lambda: (True, ""))
        monkeypatch.setattr(mgr, "_get_base_lean_path", lambda: "/base")
        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"], captured["env"] = args, kwargs.get("env")
            proc = AsyncMock()
            proc.returncode = None
            proc.stdout.readline = AsyncMock(
                return_value=(mgr.READY_SIGNAL + "\n").encode()
            )
            return proc

        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", fake_exec
        )
        assert await mgr.start()
        args = list(captured["args"])
        assert "--index-mode" in args and "use" in args and "--index-file" in args
        assert "--read-index" not in args
        assert "--path" not in args
        assert "LEAN_PATH" in captured["env"]


@pytest.mark.slow
class TestLoogleInstall:
    """Install loogle binary. Run with: pytest -m slow tests/unit/test_loogle.py

    Requires git, lake, ~2GB disk space. Takes several minutes on first run.
    """

    @pytest.mark.asyncio
    async def test_install_loogle(self):
        import shutil

        if not shutil.which("git") or not shutil.which("lake"):
            pytest.skip("git and lake required")

        mgr = LoogleManager()  # real cache dir
        assert mgr.ensure_installed(), "Failed to install loogle"
        assert mgr.is_installed


class TestLoogleQuery:
    """Test start/query/stop against installed loogle binary.

    Skips if loogle is not installed or the local cache is not runnable.
    Run TestLoogleInstall first to install or refresh the cache.
    """

    @pytest.mark.asyncio
    async def test_start_query_stop(self):
        mgr = LoogleManager()  # real cache dir
        if not mgr.is_installed:
            pytest.skip(
                "loogle not installed (run: pytest -m slow tests/unit/test_loogle.py)"
            )

        try:
            started = await mgr.start()
            if not started:
                await mgr.stop()
                pytest.skip(
                    "loogle installed but not runnable "
                    "(run: pytest -m slow tests/unit/test_loogle.py)"
                )
            assert mgr.is_running

            results = await mgr.query("Nat.add", num_results=3)
            assert len(results) > 0
            assert any("add" in r.get("name", "").lower() for r in results)
        finally:
            await mgr.stop()
            assert not mgr.is_running
