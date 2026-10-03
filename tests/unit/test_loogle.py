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
            patch.object(mgr, "_run", return_value=MagicMock(returncode=0)),
            patch.object(mgr, "_checkout_repo_ref", return_value=True),
        ):
            assert mgr._clone_repo()

    def test_clone_repo_fail(self, mgr):
        with patch.object(
            mgr, "_run", return_value=MagicMock(returncode=1, stderr="err")
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
        # Exercise the failed-start branch without downloading/building Loogle.
        with patch.object(
            mgr, "start", new_callable=AsyncMock, return_value=False
        ) as start:
            with pytest.raises(RuntimeError, match="Failed to start"):
                await mgr.query("test")
        start.assert_awaited_once()

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

    Skips only when loogle is not installed; a broken installation fails.
    Run TestLoogleInstall first to install or refresh the cache.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(360)
    async def test_start_query_stop(self):
        project = Path(__file__).resolve().parents[1] / "test_project"
        mgr = LoogleManager(project_path=project)  # real cache dir
        if not mgr.is_installed:
            pytest.skip(
                "loogle not installed (run: pytest -m slow tests/unit/test_loogle.py)"
            )

        try:
            started = await mgr.start()
            assert started, "Installed Loogle failed to become ready"
            assert mgr.is_running

            results = await mgr.query("Nat.add", num_results=3)
            assert len(results) > 0
            assert any("add" in r.get("name", "").lower() for r in results)
        finally:
            await mgr.stop()
            assert not mgr.is_running


class TestLoogleOwnedStartup:
    @pytest.fixture
    def manager(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "lean-toolchain").write_text("leanprover/lean4:v4.30.0\n")
        manager = LoogleManager(cache_dir=tmp_path / "cache", project_path=project)
        manager.binary_path.parent.mkdir(parents=True)
        manager.binary_path.touch()
        return manager

    def test_existing_install_needs_no_git_or_writable_checkout(
        self, manager, monkeypatch
    ):
        def unexpected():
            raise AssertionError(
                "Existing pinned installation was unnecessarily rebuilt"
            )

        monkeypatch.setattr(manager, "_check_prerequisites", unexpected)
        monkeypatch.setattr(manager, "_clone_repo", unexpected)
        monkeypatch.setattr(manager, "_build_loogle", unexpected)
        assert manager.ensure_installed()

    @pytest.mark.asyncio
    async def test_start_accepts_prelude_and_drains_stderr(self, manager, monkeypatch):
        import sys

        real_exec = asyncio.create_subprocess_exec

        async def launch(*args, **kwargs):
            return await real_exec(
                sys.executable,
                "-c",
                (
                    "import sys,time;sys.stderr.write('x'*262144);sys.stderr.flush();"
                    "print('loading index',flush=True);print('Loogle is ready.',flush=True);time.sleep(60)"
                ),
                **kwargs,
            )

        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", launch
        )
        try:
            assert await asyncio.wait_for(manager.start(), timeout=5)
            assert len(manager._stderr_tail) == 8192
        finally:
            await manager.stop()
        assert manager.process is None
        assert manager._stderr_task is None

    @pytest.mark.asyncio
    async def test_failure_before_ready_reaps_process_and_diagnostics(
        self, manager, monkeypatch, caplog
    ):
        import sys

        real_exec = asyncio.create_subprocess_exec
        launched = []

        async def launch(*args, **kwargs):
            process = await real_exec(
                sys.executable,
                "-c",
                "import sys;sys.stderr.write('bad index fixture\\n');sys.exit(2)",
                **kwargs,
            )
            launched.append(process)
            return process

        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", launch
        )
        assert not await manager.start()
        assert manager.process is None
        assert launched[0].returncode == 2
        assert "bad index fixture" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.name != "posix", reason="POSIX process group ownership")
    @pytest.mark.parametrize("cancel_start", [False, True])
    async def test_stop_or_cancel_owns_lake_descendant(
        self, manager, monkeypatch, tmp_path, cancel_start
    ):
        import sys
        import psutil

        real_exec = asyncio.create_subprocess_exec
        pid_path = tmp_path / "child.pid"
        child_code = "import time;time.sleep(60)"
        root_code = (
            "import subprocess,sys,time;from pathlib import Path;"
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
            f"Path({str(pid_path)!r}).write_text(str(p.pid));"
            + ("print('Loogle is ready.',flush=True);" if not cancel_start else "")
            + "time.sleep(60)"
        )

        async def launch(*args, **kwargs):
            return await real_exec(sys.executable, "-c", root_code, **kwargs)

        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", launch
        )
        start = asyncio.create_task(manager.start())
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not pid_path.exists():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)
            child_pid = int(pid_path.read_text())
            if cancel_start:
                start.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await start
            else:
                assert await start
                await manager.stop()
            assert manager.process is None
            assert manager._stderr_task is None
            assert (
                not psutil.pid_exists(child_pid)
                or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
            )
        finally:
            if not start.done():
                start.cancel()
                await asyncio.gather(start, return_exceptions=True)
            await manager.stop()

    @pytest.mark.asyncio
    async def test_cancel_start_drains_installer_thread(self, manager, monkeypatch):
        import threading

        started = threading.Event()
        finished = threading.Event()
        manager.binary_path.unlink()

        def install():
            started.set()
            assert manager._install_cancel.wait(timeout=5)
            finished.set()
            return False

        monkeypatch.setattr(manager, "ensure_installed", install)
        task = asyncio.create_task(manager.start())
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    @pytest.mark.skipif(os.name != "posix", reason="POSIX process group ownership")
    def test_installer_timeout_kills_descendants(self, manager, tmp_path):
        import sys
        import subprocess
        import psutil

        pid_path = tmp_path / "installer-child.pid"
        code = (
            "import subprocess,sys,time;from pathlib import Path;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            f"Path({str(pid_path)!r}).write_text(str(p.pid));time.sleep(60)"
        )
        with pytest.raises(subprocess.TimeoutExpired):
            manager._run([sys.executable, "-c", code], timeout=0.3, cwd=tmp_path)
        child_pid = int(pid_path.read_text())
        assert (
            not psutil.pid_exists(child_pid)
            or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )

    @pytest.mark.asyncio
    async def test_query_cancellation_discards_late_response(
        self, manager, monkeypatch, tmp_path
    ):
        import sys

        real_exec = asyncio.create_subprocess_exec
        seen = tmp_path / "query-seen"
        code = (
            "import sys,time,json;from pathlib import Path\n"
            "print('Loogle is ready.',flush=True)\n"
            "for line in sys.stdin:\n"
            f" Path({str(seen)!r}).write_text(line.strip())\n"
            " time.sleep(0.2)\n"
            " print(json.dumps({'hits':[{'name':line.strip()}]}),flush=True)\n"
        )

        async def launch(*args, **kwargs):
            return await real_exec(sys.executable, "-c", code, **kwargs)

        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", launch
        )
        assert await manager.start()
        old = manager.process
        query = asyncio.create_task(manager.query("first"))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not seen.exists():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)
            query.cancel()
            with pytest.raises(asyncio.CancelledError):
                await query
            assert manager.process is None
            assert old.returncode is not None
            result = await manager.query("second")
            assert result[0]["name"] == "second"
        finally:
            if not query.done():
                query.cancel()
                await asyncio.gather(query, return_exceptions=True)
            await manager.stop()

    @pytest.mark.asyncio
    async def test_overlapping_stop_and_repeated_cancellation_share_cleanup(
        self, manager
    ):
        entered = asyncio.Event()
        release = asyncio.Event()
        proc = MagicMock()
        proc.returncode = None

        async def wait():
            entered.set()
            await release.wait()
            proc.returncode = 0
            return 0

        proc.wait = AsyncMock(side_effect=wait)
        manager.process, manager._ready = proc, True
        first = asyncio.create_task(manager.stop())
        await entered.wait()
        second = asyncio.create_task(manager.stop())
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        proc.terminate.assert_called_once()
        proc.wait.assert_awaited_once()
        assert manager._stop_task is None

    @pytest.mark.asyncio
    async def test_stop_during_pending_spawn_cannot_resurrect_server(
        self, manager, monkeypatch
    ):
        import sys

        real_exec = asyncio.create_subprocess_exec
        spawn_entered = asyncio.Event()
        release = asyncio.Event()
        processes = []

        async def launch(*args, **kwargs):
            spawn_entered.set()
            await release.wait()
            process = await real_exec(
                sys.executable,
                "-c",
                "import time;print('Loogle is ready.',flush=True);time.sleep(60)",
                **kwargs,
            )
            processes.append(process)
            return process

        monkeypatch.setattr(
            "lean_lsp_mcp.loogle.asyncio.create_subprocess_exec", launch
        )
        start = asyncio.create_task(manager.start())
        try:
            await spawn_entered.wait()
            await manager.stop()
            release.set()
            assert not await start
            assert not manager.is_running
            assert manager.process is None
            assert processes[0].returncode is not None
        finally:
            release.set()
            if not start.done():
                start.cancel()
                await asyncio.gather(start, return_exceptions=True)
            await manager.stop()

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.name != "posix", reason="POSIX process group ownership")
    async def test_ephemeral_macos_group_error_does_not_skip_reaping(
        self, manager, monkeypatch
    ):
        from lean_lsp_mcp import loogle

        proc = MagicMock(pid=12345)
        proc.returncode = None
        proc.wait = AsyncMock(return_value=0)
        member = MagicMock()
        member.children.return_value = []
        member.is_running.return_value = False
        monkeypatch.setattr(loogle.psutil, "Process", lambda _: member)
        calls = []

        def signal_group(group, sig):
            calls.append(sig)
            if sig == 0:
                raise PermissionError("dying group")

        monkeypatch.setattr(loogle.os, "killpg", signal_group)
        manager.process = proc
        manager._process_group = proc.pid
        await manager.stop()
        proc.wait.assert_awaited_once()
        assert manager.process is None
        assert len(calls) == 2

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.name != "posix", reason="POSIX process group ownership")
    async def test_macos_dying_group_poll_waits_for_owned_member_exit(
        self, manager, monkeypatch
    ):
        from lean_lsp_mcp import loogle

        proc = MagicMock(pid=12345)
        proc.returncode = None
        proc.wait = AsyncMock(return_value=0)
        member = MagicMock()
        member.children.return_value = []
        member.is_running.side_effect = [True, False]
        member.status.return_value = "idle"
        monkeypatch.setattr(loogle.psutil, "Process", lambda _: member)
        polls = []

        def signal_group(group, sig):
            if sig == 0:
                polls.append(sig)
                raise PermissionError("native exit teardown")

        monkeypatch.setattr(loogle.os, "killpg", signal_group)
        manager.process, manager._process_group = proc, proc.pid
        await manager.stop()
        assert len(polls) == 2
        proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_external_stop_cancels_and_drains_install_thread(
        self, manager, monkeypatch
    ):
        import threading

        started, finished = threading.Event(), threading.Event()
        manager.binary_path.unlink()

        def install():
            started.set()
            assert manager._install_cancel.wait(timeout=5)
            finished.set()
            return False

        monkeypatch.setattr(manager, "ensure_installed", install)
        start = asyncio.create_task(manager.start())
        while not started.is_set():
            await asyncio.sleep(0.01)
        await manager.stop()
        assert finished.is_set()
        assert not await start
        assert manager._install_task is None
