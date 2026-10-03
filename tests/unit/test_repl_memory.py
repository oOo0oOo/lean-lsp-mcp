"""Memory protection without relying on Darwin's misleading RLIMIT_RSS alias."""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from lean_lsp_mcp import repl as module
from lean_lsp_mcp.repl import Repl, _memory_limit_preexec


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_no_address_space_preexec_on_non_linux(monkeypatch, system):
    monkeypatch.setattr(module.platform, "system", lambda: system)
    assert _memory_limit_preexec(16384) is None


@pytest.mark.skipif(sys.platform == "win32", reason="resource is Unix-only")
def test_linux_respects_lower_inherited_hard_limit(monkeypatch):
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    cap = 128 * 1024**2
    monkeypatch.setattr(module.resource, "getrlimit", lambda _: (cap, cap))
    setter = Mock()
    monkeypatch.setattr(module.resource, "setrlimit", setter)
    _memory_limit_preexec(16384)()
    setter.assert_called_once_with(module.resource.RLIMIT_AS, (cap, cap))


async def test_memory_guard_counts_children_and_reports_limit(monkeypatch, tmp_path):
    repl = Repl(str(tmp_path))
    repl.mem_mb = 10
    child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=6 * 1024**2))
    process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=5 * 1024**2),
        children=lambda recursive: [child],
    )
    monkeypatch.setattr(module.psutil, "Process", lambda _: process)
    kill = Mock()
    monkeypatch.setattr(repl, "_kill_process", kill)
    proc = SimpleNamespace(pid=123, returncode=None, stderr=None)
    await repl._watch_memory(proc)
    kill.assert_called_once_with(proc)
    repl._proc = proc
    proc.returncode = -9
    proc.wait = lambda: asyncio.sleep(0)
    error = await repl._process_error("No response")
    assert "exceeded resident memory limit of 10 MiB" in str(error)


async def test_memory_guard_stops_when_measurement_is_denied(monkeypatch, tmp_path):
    repl = Repl(str(tmp_path))
    monkeypatch.setattr(
        module.psutil, "Process", Mock(side_effect=psutil.AccessDenied(123))
    )
    kill = Mock()
    monkeypatch.setattr(repl, "_kill_process", kill)
    proc = SimpleNamespace(pid=123, returncode=None)
    await repl._watch_memory(proc)
    kill.assert_called_once_with(proc)
    assert "Cannot monitor REPL" in repl._memory_error


async def test_memory_guard_tolerates_exited_process(monkeypatch, tmp_path):
    repl = Repl(str(tmp_path))
    kill = Mock()
    monkeypatch.setattr(repl, "_kill_process", kill)
    monkeypatch.setattr(
        module.psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(123))
    )
    proc = SimpleNamespace(pid=123, returncode=None)
    await repl._watch_memory(proc)
    kill.assert_called_once_with(proc)
    assert repl._memory_error is None


@pytest.mark.skipif(os.name != "posix", reason="new-session process group required")
async def test_real_memory_guard_kills_and_reaps_process(tmp_path):
    repl = Repl(str(tmp_path))
    repl.mem_mb = 32
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; data = bytearray(64*1024**2); time.sleep(30)",
        start_new_session=True,
    )
    repl._proc = proc
    monitor = asyncio.create_task(repl._watch_memory(proc))
    repl._memory_task = monitor
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
        await monitor
        assert proc.returncode != 0
        assert "exceeded resident memory" in repl._memory_error
    finally:
        await repl.close()
    assert repl._memory_task is None
    assert repl._proc is None


@pytest.mark.skipif(os.name != "posix", reason="new-session process group required")
async def test_close_drains_healthy_memory_guard(tmp_path):
    repl = Repl(str(tmp_path))
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(30)", start_new_session=True
    )
    repl._proc = proc
    monitor = asyncio.create_task(repl._watch_memory(proc))
    repl._memory_task = monitor
    try:
        await asyncio.sleep(0.05)
        assert not monitor.done()
    finally:
        await repl.close()
    assert monitor.done()
    assert repl._memory_task is None
    assert proc.returncode is not None


@pytest.mark.skipif(os.name != "posix", reason="new-session process group required")
async def test_memory_guard_cleans_children_after_parent_exits(tmp_path):
    repl = Repl(str(tmp_path))
    pid_file = tmp_path / "child.pid"
    parent_code = (
        "import subprocess, sys; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "Path(sys.argv[1]).write_text(str(child.pid))"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", parent_code, str(pid_file), start_new_session=True
    )
    repl._proc = proc
    child = None
    try:
        await asyncio.wait_for(proc.wait(), 10)
        child = psutil.Process(int(pid_file.read_text()))
        monitor = asyncio.create_task(repl._watch_memory(proc))
        repl._memory_task = monitor
        await asyncio.wait_for(monitor, 2)
        for _ in range(100):
            if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("REPL descendant survived its parent's watchdog shutdown")
    finally:
        await repl.close()
        if child is not None:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass


async def test_same_header_restart_drains_previous_process(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    repl = Repl(str(tmp_path))
    repl._header = "import Mathlib"
    repl._header_env = 1
    repl._proc = SimpleNamespace(returncode=0)
    order = []

    async def close():
        order.append("close")
        repl._proc = None
        repl._header = None

    async def start():
        order.append("start")
        assert repl._header == "import Mathlib"

    monkeypatch.setattr(repl, "close", close)
    monkeypatch.setattr(repl, "_start", start)
    monkeypatch.setattr(repl, "_send_cmd", AsyncMock(return_value={"env": 2}))
    assert await repl._ensure_header("import Mathlib") == 2
    assert order == ["close", "start"]
