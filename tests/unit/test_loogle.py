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
        project = tmp_path / "project"
        project.mkdir()
        (project / "lean-toolchain").write_text("leanprover/lean4:v4.30.0\n")
        return LoogleManager(cache_dir=tmp_path / "loogle", project_path=project)

    def test_binary_path(self, mgr):
        assert mgr.binary_path == mgr.build_dir / "bin" / "loogle"
        assert mgr.repo_dir.name.startswith(f"repo-{mgr.REPO_REF[:12]}-")

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
        (mgr.repo_dir / ".git").mkdir(parents=True)
        with patch.object(mgr, "_checkout_repo_ref", return_value=True):
            assert mgr._clone_repo()

    def test_clone_repo_rejects_non_git_directory(self, mgr):
        mgr.repo_dir.mkdir(parents=True)
        assert not mgr._clone_repo()

    def test_checkout_repo_ref_reuses_pinned_revision(self, mgr, monkeypatch):
        calls = []

        def fake_run(cmd, timeout=300, cwd=None, env=None):
            calls.append(cmd)
            return MagicMock(returncode=0, stdout=f"{mgr.REPO_REF}\n", stderr="")

        monkeypatch.setattr(mgr, "_run", fake_run)
        assert mgr._checkout_repo_ref()
        assert calls == [
            ["git", "rev-parse", "HEAD"],
            ["git", "checkout", "--detach", mgr.REPO_REF],
        ]

    def test_checkout_repo_ref_fetches_pinned_revision(self, mgr, monkeypatch):
        calls = []

        def fake_run(cmd, timeout=300, cwd=None, env=None):
            calls.append(cmd)
            stdout = "old\n" if "rev-parse" in cmd else ""
            return MagicMock(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(mgr, "_run", fake_run)
        assert mgr._checkout_repo_ref()
        assert calls[1] == [
            "git",
            "fetch",
            "--depth",
            "1",
            "origin",
            mgr.REPO_REF,
        ]
        assert calls[2] == ["git", "checkout", "--detach", mgr.REPO_REF]

    def test_clone_repo_success(self, mgr):
        with (
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch.object(mgr, "_checkout_repo_ref", return_value=True),
        ):
            assert mgr._clone_repo()

    def test_clone_repo_fail(self, mgr):
        with patch(
            "subprocess.run", return_value=MagicMock(returncode=1, stderr="err")
        ):
            assert not mgr._clone_repo()

    def test_project_toolchain_controls_build_dir(self, tmp_path):
        project1 = tmp_path / "project1"
        project2 = tmp_path / "project2"
        project1.mkdir()
        project2.mkdir()
        (project1 / "lean-toolchain").write_text("leanprover/lean4:v4.29.0")
        (project2 / "lean-toolchain").write_text("leanprover/lean4:v4.30.0")
        cache = tmp_path / "cache"

        first = LoogleManager(cache_dir=cache, project_path=project1)
        second = LoogleManager(cache_dir=cache, project_path=project2)

        assert first.build_dir != second.build_dir
        assert first.binary_path != second.binary_path

    def test_check_environment(self, mgr):
        ok, msg = mgr.check_environment()
        assert not ok
        assert "binary not found" in msg

        mgr.binary_path.parent.mkdir(parents=True)
        mgr.binary_path.touch()
        assert mgr.check_environment() == (True, "")

    def test_check_environment_requires_project_toolchain(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        mgr = LoogleManager(cache_dir=tmp_path / "cache", project_path=project)
        ok, msg = mgr.check_environment()
        assert not ok
        assert "lean-toolchain" in msg

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
        project = tmp_path / "project"
        project.mkdir()
        (project / "lean-toolchain").write_text("leanprover/lean4:v4.30.0")
        mgr = LoogleManager(cache_dir=tmp_path / "loogle", project_path=project)
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

    def test_get_project_toolchain(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        mgr = LoogleManager(cache_dir=tmp_path / "cache", project_path=project)
        assert mgr._get_project_toolchain() is None
        (project / "lean-toolchain").write_text("leanprover/lean4:v4.28.0\n")
        assert mgr._get_project_toolchain() == "leanprover/lean4:v4.28.0"

    def test_get_project_toolchain_no_project(self, tmp_path):
        mgr = LoogleManager(cache_dir=tmp_path / "cache")
        assert mgr._get_project_toolchain() is None

    def test_index_path_is_project_specific(self, tmp_path):
        cache = tmp_path / "cache"
        first_project = tmp_path / "first"
        second_project = tmp_path / "second"
        first_project.mkdir()
        second_project.mkdir()

        first = LoogleManager(cache_dir=cache, project_path=first_project)
        second = LoogleManager(cache_dir=cache, project_path=second_project)

        assert first.index_path != second.index_path
        assert first.index_path.parent == second.index_path.parent == cache / "index"

    def test_set_project_path(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()

        mgr = LoogleManager(cache_dir=tmp_path / "cache")
        assert mgr.set_project_path(project)
        assert mgr.project_path == project.resolve()
        assert not mgr.set_project_path(project)

    def test_project_env_uses_project_toolchain(self, mgr):
        env = mgr._project_env()
        assert "LEAN_PATH" not in env
        assert env["LAKE_ARTIFACT_CACHE"] == "false"
        assert env["ELAN_TOOLCHAIN"] == "leanprover/lean4:v4.30.0"

    def test_build_uses_project_toolchain_and_versioned_build_dir(
        self, mgr, monkeypatch
    ):
        mgr.repo_dir.mkdir(parents=True)
        captured = {}

        def fake_run(cmd, timeout=300, cwd=None, env=None):
            captured["cmd"], captured["env"] = cmd, env
            mgr.binary_path.parent.mkdir(parents=True)
            mgr.binary_path.touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(mgr, "_run", fake_run)
        assert mgr._build_loogle()
        assert captured["cmd"] == ["lake", "build", "loogle"]
        assert captured["env"]["ELAN_TOOLCHAIN"] == "leanprover/lean4:v4.30.0"
        assert "LEAN_PATH" not in captured["env"]

    @pytest.mark.asyncio
    async def test_start_uses_lake_env_and_native_defaults(self, mgr, monkeypatch):
        mgr.binary_path.parent.mkdir(parents=True)
        mgr.binary_path.touch()
        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"], captured["kwargs"] = args, kwargs
            proc = AsyncMock()
            proc.returncode = None
            proc.stdout.readline = AsyncMock(
                return_value=(mgr.READY_SIGNAL + "\n").encode()
            )
            proc.stderr.read = AsyncMock(return_value=b"")
            return proc

        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", fake_exec
        )
        assert await mgr.start()
        args = list(captured["args"])
        assert args == [
            "lake",
            "env",
            str(mgr.binary_path),
            "--json",
            "--interactive",
            "--index-file",
            str(mgr.index_path),
        ]
        assert captured["kwargs"]["cwd"] == mgr.project_path
        assert "LEAN_PATH" not in captured["kwargs"]["env"]
        assert captured["kwargs"]["env"]["ELAN_TOOLCHAIN"] == "leanprover/lean4:v4.30.0"

    @pytest.mark.asyncio
    async def test_start_builds_binary_for_new_project_toolchain(
        self, mgr, monkeypatch
    ):
        calls = []

        def fake_install():
            calls.append("install")
            mgr.binary_path.parent.mkdir(parents=True)
            mgr.binary_path.touch()
            return True

        async def fake_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.returncode = None
            proc.stdout.readline = AsyncMock(
                return_value=(mgr.READY_SIGNAL + "\n").encode()
            )
            proc.stderr.read = AsyncMock(return_value=b"")
            return proc

        monkeypatch.setattr(mgr, "ensure_installed", fake_install)
        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", fake_exec
        )

        assert await mgr.start()
        assert calls == ["install"]


@pytest.mark.slow
class TestLoogleInstall:
    """Install loogle binary. Run with: pytest -m slow tests/unit/test_loogle.py

    Requires git and lake. The first Mathlib index build takes several minutes.
    """

    @pytest.mark.asyncio
    async def test_install_loogle(self):
        import shutil

        if not shutil.which("git") or not shutil.which("lake"):
            pytest.skip("git and lake required")

        project = Path(__file__).resolve().parents[1] / "test_project"
        mgr = LoogleManager(project_path=project)  # real cache dir
        assert mgr.ensure_installed(), "Failed to install loogle"
        assert mgr.is_installed


class TestLoogleQuery:
    """Test start/query/stop against installed loogle binary.

    Skips if loogle is not installed or the local cache is not runnable.
    Run TestLoogleInstall first to install or refresh the cache.
    """

    @pytest.mark.asyncio
    async def test_start_query_stop(self):
        project = Path(__file__).resolve().parents[1] / "test_project"
        mgr = LoogleManager(project_path=project)  # real cache dir
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
